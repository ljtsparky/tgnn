"""
infer_wan.py
============
Graph-conditioned video generation with Wan2.1-T2V-1.3B + WanGraphAdapterV2.

Pipeline overview
-----------------
1. Load frozen Wan2.1 pipeline (VAE + UMT5 text encoder + transformer)
2. Load trained TGNNEncoder + WanGraphAdapterV2 from checkpoint
3. Pick a test scene graph from Action Genome
4. TGNNEncoder encodes graph nodes → node_emb [N, 256]
5. Pad node embeddings → node_tokens [1, 32, 256]
6. Register node_tokens in adapter (hook fires inside transformer.condition_embedder)
7. WanPipeline generates video: at each denoising step, the hook injects
   graph_tokens [1, 32, 1536] BEFORE the transformer blocks see encoder_hs,
   so all 30 cross-attention blocks attend to [graph_tokens | text_tokens]

Usage
-----
    python infer_wan.py --checkpoint .../checkpoints_wan_v2/step_005000
    python infer_wan.py --checkpoint ... --video_id AAAA5
"""

from __future__ import annotations

import argparse, os, random, sys
from pathlib import Path

import torch
from torch_geometric.data import Batch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model     import TGNNEncoder
from adapter_wan_v2 import WanGraphAdapterV2
from adapter        import pad_node_embeddings
from train_wan      import encode_prompt_wan

GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
CKPT_DIR  = "/projects/bgnv/leatherman/tgnn/checkpoints_wan_v2/step_005000"
HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
OUT_DIR   = "/projects/bgnv/leatherman/tgnn/outputs/wan"

N_GRAPH_TOKENS = 32   # number of node tokens padded/truncated to per graph
D_GRAPH        = 256  # TGNNEncoder output dimension per node


def pick_graph(graph_dir, video_id, split="test"):
    """Load a specific graph by video_id, or pick randomly from split."""
    graph_dir = Path(graph_dir)
    if video_id:
        return torch.load(graph_dir / f"{video_id}.pt", weights_only=False)
    candidates = [p for p in sorted(graph_dir.glob("*.pt"))
                  if torch.load(p, weights_only=False).split == split]
    pt = random.choice(candidates)
    print(f"  Selected: {pt.stem}")
    return torch.load(pt, weights_only=False)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}")

    # ── [1] Load Wan2.1 pipeline (all components frozen) ────────────────────
    # Components: VAE (3D causal), UMT5 text encoder, WanTransformer3DModel
    # Resolution: 480×832 (16:9), Frames: 17 (16N+1 format, ~1s at 16fps)
    print("\n[1] Loading Wan2.1 pipeline ...")
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    pipe.vae.enable_slicing()   # process VAE temporal slices to save memory
    pipe.vae.enable_tiling()    # process VAE spatial tiles to save memory

    # d_inner: transformer inner dimension = n_heads × head_dim = 12 × 128 = 1536
    d_inner = pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim
    print(f"  Transformer: {len(pipe.transformer.blocks)} blocks, inner_dim={d_inner}")

    # ── [2] Load trained adapter checkpoint ─────────────────────────────────
    # TGNNEncoder:      796K params, kept in fp32 (GATv2Conv bf16 compat issue)
    # WanGraphAdapterV2: 3.55M params, bf16 to match pipeline
    # Hook registered on transformer.condition_embedder at adapter init
    print(f"\n[2] Loading checkpoint: {args.checkpoint}")
    tgnn = TGNNEncoder(d_model=D_GRAPH).to(device).eval()
    # fp32 intentionally — GATv2Conv has bfloat16 compatibility issues
    adapter = WanGraphAdapterV2(
        transformer    = pipe.transformer,
        d_graph        = D_GRAPH,
        n_graph_tokens = N_GRAPH_TOKENS,
        d_inner        = d_inner,
    ).to(device=device, dtype=dtype).eval()

    tgnn.load_state_dict(torch.load(
        Path(args.checkpoint) / "tgnn.pt",
        map_location=device, weights_only=True,
    ))
    adapter.load_state_dict(torch.load(
        Path(args.checkpoint) / "adapter.pt",
        map_location=device, weights_only=True,
    ))
    adapter.to(dtype)   # re-cast after load_state_dict (restores fp32 → bf16)
    print("  Checkpoint loaded.")

    # ── [3] Load scene graph ─────────────────────────────────────────────────
    print("\n[3] Loading scene graph ...")
    graph = pick_graph(args.graph_dir, args.video_id, args.split)
    print(f"  Video ID : {graph.video_id}")
    print(f"  Nodes    : {graph.num_nodes}  Edges: {graph.num_edges}")
    # node_class: [N] long  — object category index (0=person, 1-35=objects)
    # edge_attr:  [E, 28]   — attention + spatial + contacting + IoU features
    # edge_type:  [E]       — 0=spatial (within frame), 1=temporal (across frames)

    # ── [4] Encode graph → node tokens ──────────────────────────────────────
    print("\n[4] Encoding graph ...")
    graph_batch = Batch.from_data_list([graph]).to(device)

    with torch.no_grad():
        # TGNNEncoder: 3× GATv2Conv + AttentionalAggregation
        # node_emb: [N_total, 256]  — per-node embeddings (variable N per video)
        # graph_emb: [1, 256]       — pooled graph-level embedding (not used here)
        node_emb, _ = tgnn(graph_batch)

        # Pad/truncate to fixed N_GRAPH_TOKENS per graph
        # node_tokens: [1, 32, 256]  — batch of padded node embeddings
        node_tokens = pad_node_embeddings(
            node_emb, graph_batch.batch, N_GRAPH_TOKENS, D_GRAPH,
        ).to(dtype)

        # Store in adapter — hook will inject these into transformer
        # during each forward pass of the denoising loop
        adapter.set_graph_tokens(node_tokens)
        # node_tokens: [1, 32, 256] → inside hook: projected to [1, 32, 1536]
        #              → concatenated with encoder_hs [1, 512, 1536]
        #              → all 30 blocks see [1, 544, 1536] in cross-attention

    print(f"  node_emb    : {node_emb.shape}   (N_nodes × 256)")
    print(f"  node_tokens : {node_tokens.shape}  (1 × 32 × 256, will be injected as 1536-dim)")

    # ── [5] Generate video ───────────────────────────────────────────────────
    # The pipeline handles: text encoding → VAE noise init → 50 denoising steps
    # At each step: transformer.forward() → condition_embedder fires hook
    # Hook injects graph_tokens [1, 32, 1536] into encoder_hs [1, 512, 1536]
    # Result: all cross-attention blocks see 544 tokens (32 graph + 512 text)
    print(f"\n[5] Generating ({args.num_frames} frames, {args.steps} steps, "
          f"cfg={args.guidance_scale}, {args.height}×{args.width}) ...")

    with torch.no_grad():
        output = pipe(
            prompt              = "a person in an indoor scene",
            negative_prompt     = "",
            height              = args.height,
            width               = args.width,
            num_frames          = args.num_frames,
            num_inference_steps = args.steps,
            guidance_scale      = args.guidance_scale,
            generator           = torch.Generator(device=device).manual_seed(args.seed),
        )
    # output.frames[0]: list of 17 PIL images, each 480×832

    # Clear hook state (important: prevents graph from leaking into next call)
    adapter.clear_graph_tokens()

    # ── [6] Save ─────────────────────────────────────────────────────────────
    frames   = output.frames[0]   # list[PIL.Image], len=num_frames
    out_path = Path(args.out_dir) / f"{graph.video_id}_wan_graph_cond_v2.mp4"
    export_to_video(frames, str(out_path), fps=16)
    print(f"\n[6] Saved → {out_path}")
    print(f"    Frames: {len(frames)}, Resolution: {args.width}×{args.height}, FPS: 16")


def parse_args():
    p = argparse.ArgumentParser(description="Graph-conditioned Wan2.1 inference")
    p.add_argument("--checkpoint",     default=CKPT_DIR)
    p.add_argument("--graph_dir",      default=GRAPH_DIR)
    p.add_argument("--video_id",       default=None)
    p.add_argument("--split",          default="test")
    p.add_argument("--hf_cache",       default=HF_CACHE)
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--height",         type=int,   default=480)
    p.add_argument("--width",          type=int,   default=832)
    p.add_argument("--num_frames",     type=int,   default=17)
    p.add_argument("--steps",          type=int,   default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
