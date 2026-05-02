"""
infer_sdxl.py
=============
Generate graph-conditioned images with SDXL + trained TGNN/Adapter.

For each test graph, we randomly pick a target frame from frame_ids and
generate a 1024×1024 image. Output is saved with both graph-cond and
text-only baseline (single pipeline reload, two passes per graph).
"""

from __future__ import annotations

import argparse, os, random, sys
from pathlib import Path

import torch
from torch_geometric.data import Batch
from diffusers import StableDiffusionXLPipeline

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model    import TGNNEncoder
from adapter_sdxl  import SDXLGraphAdapter, pad_node_embeddings_temporal
from dataset_image import build_prompt_from_graph

MODEL_ID  = "stabilityai/stable-diffusion-xl-base-1.0"
HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"


def list_test_graphs(graph_dir, split="test", n=None, seed=42):
    cands = []
    for p in sorted(Path(graph_dir).glob("*.pt")):
        try:
            g = torch.load(p, weights_only=False)
            if g.split == split:
                cands.append(p)
        except Exception:
            pass
    rng = random.Random(seed); rng.shuffle(cands)
    return cands[:n] if n else cands


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache
    device = torch.device("cuda")
    dtype  = torch.bfloat16

    print("[1] Loading SDXL ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    # VAE decode in bf16 is stable (only encode of certain images NaNs in fp16);
    # bf16 has fp32-equivalent exponent range — keep it bf16 for inference.
    cross_dim = pipe.unet.config.cross_attention_dim

    print("[2] Loading TGNN + adapter ...")
    tgnn    = TGNNEncoder(d_model=256).to(device).eval()
    adapter = SDXLGraphAdapter(d_graph=256, cross_dim=cross_dim).to(device, dtype).eval()
    adapter.install(pipe.unet)

    tgnn.load_state_dict(torch.load(
        Path(args.checkpoint) / "tgnn.pt", map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(
        Path(args.checkpoint) / "adapter.pt", map_location=device, weights_only=True))
    adapter.to(dtype)
    print(f"  Loaded {args.checkpoint}")

    graphs = list_test_graphs(args.graph_dir, "test", args.n, args.seed)
    print(f"\n[3] Generating {len(graphs)} graph + baseline pairs ...")

    for i, gpath in enumerate(graphs):
        g = torch.load(gpath, weights_only=False)
        # Pick mid frame as target
        T = int(g.num_frames)
        target_idx = T // 2
        # Object-aware prompt — same for baseline AND graph-cond (fair comparison;
        # text gives concept words, graph adds spatial precision).
        prompt = build_prompt_from_graph(g, target_idx)
        print(f"\n[{i+1}/{len(graphs)}] {g.video_id} | nodes={g.num_nodes} edges={g.num_edges} "
              f"target_frame={int(g.frame_ids[target_idx])}")
        print(f"  prompt: {prompt}")

        graph_batch = Batch.from_data_list([g]).to(device)
        with torch.no_grad():
            tgt = torch.tensor([target_idx], device=device, dtype=torch.long)
            node_emb, _, is_target = tgnn(graph_batch, tgt)
            node_tokens = pad_node_embeddings_temporal(
                node_emb, graph_batch.batch, graph_batch.node_frame, tgt,
                is_target, n_tokens=args.n_graph_tokens, d_graph=256,
            ).to(dtype)
            graph_tokens = adapter.encode(node_tokens)

        # ── Graph-cond pass ───────────────────────────────────────
        adapter.set_graph_tokens(graph_tokens)
        out_g = pipe(
            prompt=prompt, height=1024, width=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.Generator("cpu").manual_seed(args.seed + i),
        )
        adapter.clear_graph_tokens()
        out_g.images[0].save(Path(args.out_dir) / f"{g.video_id}_graph_cond.png")

        # ── Baseline (no graph) pass — same prompt ────────────────
        adapter.set_graph_tokens(None)
        out_b = pipe(
            prompt=prompt, height=1024, width=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            generator=torch.Generator("cpu").manual_seed(args.seed + i),
        )
        out_b.images[0].save(Path(args.out_dir) / f"{g.video_id}_baseline.png")
        print(f"  saved → {g.video_id}_graph_cond.png + _baseline.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     required=True)
    p.add_argument("--graph_dir",      default=GRAPH_DIR)
    p.add_argument("--hf_cache",       default=HF_CACHE)
    p.add_argument("--out_dir",        default="/projects/bgnv/leatherman/tgnn/outputs/sdxl")
    p.add_argument("--n",              type=int, default=20)
    p.add_argument("--steps",          type=int, default=30)
    p.add_argument("--cfg",            type=float, default=7.5)
    p.add_argument("--n_graph_tokens", type=int, default=64)
    p.add_argument("--seed",           type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
