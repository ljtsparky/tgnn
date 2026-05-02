"""
infer_hunyuan.py
================
Graph-conditioned video generation with HunyuanVideo + TGNNEncoder + HunyuanGraphAdapter.

Pipeline
--------
1. Load frozen HunyuanVideo pipeline (VAE + LLaMA + CLIP + Transformer)
2. Load trained TGNNEncoder + HunyuanGraphAdapter from checkpoint
3. Pick test scene graph
4. Encode graph: TGNNEncoder → node_emb [N, 256] → pad → [1, 32, 256]
5. Encode text: LLaMA → prompt_embeds [1, seq, 4096]
6. Inject graph: adapter prepends graph tokens → [1, 32+seq, 4096]
7. Generate with HunyuanVideoPipeline(prompt_embeds=combined, ...)

No hooks needed — injection done by directly modifying prompt_embeds.
"""

from __future__ import annotations

import argparse, os, random, sys
from pathlib import Path

import torch
from torch_geometric.data import Batch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model      import TGNNEncoder
from adapter_hunyuan import HunyuanGraphAdapter
from adapter         import pad_node_embeddings

MODEL_ID  = "hunyuanvideo-community/HunyuanVideo"
GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
CKPT_DIR  = "/projects/bgnv/leatherman/tgnn/checkpoints_hunyuan/step_005000"
HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
OUT_DIR   = "/projects/bgnv/leatherman/tgnn/outputs/hunyuan"
PROMPT    = "a person in an indoor scene"

N_GRAPH_TOKENS = 32
D_GRAPH        = 256


def pick_graph(graph_dir, video_id, split="test"):
    graph_dir = Path(graph_dir)
    if video_id:
        return torch.load(graph_dir / f"{video_id}.pt", weights_only=False)
    candidates = [p for p in sorted(graph_dir.glob("*.pt"))
                  if torch.load(p, weights_only=False).split == split]
    pt = random.choice(candidates)
    print(f"  Selected: {pt.stem}")
    return torch.load(pt, weights_only=False)


def list_test_graphs(graph_dir, split="test", n=None, seed=42):
    graph_dir = Path(graph_dir)
    candidates = []
    for p in sorted(graph_dir.glob("*.pt")):
        try:
            g = torch.load(p, weights_only=False)
            if g.split == split:
                candidates.append(p)
        except Exception:
            pass
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if n is not None:
        candidates = candidates[:n]
    return candidates


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── [1] Load HunyuanVideo pipeline ──────────────────────────────────────
    # transformer in bf16; rest of pipeline in fp16 for compatibility
    print("\n[1] Loading HunyuanVideo pipeline ...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer",
        torch_dtype=torch.bfloat16, local_files_only=True,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        MODEL_ID, transformer=transformer,
        torch_dtype=torch.float16, local_files_only=True,
    )
    pipe.enable_sequential_cpu_offload()   # peak ~35GB → fits on H200 80GB
    pipe.vae.enable_tiling()
    print(f"  Transformer: 20+40={len(pipe.transformer.transformer_blocks)+len(pipe.transformer.single_transformer_blocks)} blocks, "
          f"d_inner={pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim}")

    # ── [2] Load checkpoint ──────────────────────────────────────────────────
    print(f"\n[2] Loading checkpoint: {args.checkpoint}")
    tgnn    = TGNNEncoder(d_model=D_GRAPH).to(device).eval()  # keep fp32 for GATv2Conv
    adapter = HunyuanGraphAdapter(
        d_graph=D_GRAPH, d_text=4096, n_graph_tokens=N_GRAPH_TOKENS,
    ).to(device=device, dtype=torch.bfloat16).eval()

    tgnn.load_state_dict(torch.load(
        Path(args.checkpoint) / "tgnn.pt", map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(
        Path(args.checkpoint) / "adapter.pt", map_location=device, weights_only=True))
    adapter.to(torch.bfloat16)  # re-cast after load_state_dict
    print("  Checkpoint loaded.")

    # ── [3] Pick graphs to generate ─────────────────────────────────────────
    if args.video_id:
        graph_paths = [Path(args.graph_dir) / f"{args.video_id}.pt"]
    else:
        graph_paths = list_test_graphs(
            args.graph_dir, split=args.split, n=args.n_videos, seed=args.seed,
        )
    print(f"\n[3] Will generate {len(graph_paths)} videos from split={args.split}")

    # ── [4] Encode text once (same generic prompt for all) ──────────────────
    print("\n[4] Encoding text (shared across all videos) ...")
    with torch.no_grad():
        (prompt_embeds, pooled_prompt_embeds,
         prompt_attention_mask) = pipe.encode_prompt(
            prompt=PROMPT,
            max_sequence_length=args.max_seq_len,
            device=device,
        )
    print(f"  prompt_embeds: {prompt_embeds.shape}")

    # ── [5] Loop over graphs ────────────────────────────────────────────────
    for i, gpath in enumerate(graph_paths):
        graph = torch.load(gpath, weights_only=False)
        print(f"\n[{i+1}/{len(graph_paths)}] Video: {graph.video_id} "
              f"| Nodes: {graph.num_nodes} | Edges: {graph.num_edges}")
        graph_batch = Batch.from_data_list([graph]).to(device)

        out_path = Path(args.out_dir) / f"{graph.video_id}_hunyuan_graph_cond.mp4"
        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {out_path.name} exists")
            continue

        with torch.no_grad():
            node_emb, _ = tgnn(graph_batch)
            node_tokens = pad_node_embeddings(
                node_emb, graph_batch.batch, N_GRAPH_TOKENS, D_GRAPH,
            ).to(torch.bfloat16)
            combined_embeds, combined_mask = adapter(
                node_tokens,
                prompt_embeds.to(torch.bfloat16),
                prompt_attention_mask,
            )
            output = pipe(
                prompt_embeds          = combined_embeds.to(torch.float16),
                pooled_prompt_embeds   = pooled_prompt_embeds,
                prompt_attention_mask  = combined_mask,
                height                 = args.height,
                width                  = args.width,
                num_frames             = args.num_frames,
                num_inference_steps    = args.steps,
                guidance_scale         = args.guidance_scale,
                true_cfg_scale         = 1.0,
                generator              = torch.Generator("cpu").manual_seed(args.seed + i),
            )
        frames = output.frames[0]
        export_to_video(frames, str(out_path), fps=args.fps)
        print(f"  Saved → {out_path}  ({len(frames)} frames)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default=CKPT_DIR)
    p.add_argument("--graph_dir",      default=GRAPH_DIR)
    p.add_argument("--video_id",       default=None)
    p.add_argument("--n_videos",       type=int,   default=10)
    p.add_argument("--split",          default="test")
    p.add_argument("--hf_cache",       default=HF_CACHE)
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--height",         type=int,   default=480)
    p.add_argument("--width",          type=int,   default=848)
    p.add_argument("--num_frames",     type=int,   default=17)
    p.add_argument("--steps",          type=int,   default=30)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--max_seq_len",    type=int,   default=256)
    p.add_argument("--fps",            type=int,   default=16)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--overwrite",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
