"""
infer_graph.py
==============
Given a trained TGNNEncoder + GraphCondAdapter checkpoint, generate a video
conditioned on a scene graph from the Action Genome test split.

Pipeline
--------
  graph (.pt)  →  TGNNEncoder  →  graph_emb [1, 256]
  text prompt  →  T5           →  text_emb  [1, 226, 4096]
                                         ↓ FiLM adapter
                               cond [1, 226, 4096]
                                         ↓
                          CogVideoX DDIM sampling (50 steps)
                                         ↓
                               video frames → .mp4

CFG: positive = FiLM(graph_emb, text_emb)
     negative = encode("")  (unconditional, no FiLM)

Usage
-----
    # Random test video:
    python infer_graph.py --checkpoint /projects/.../checkpoints/step_005000

    # Specific video by ID:
    python infer_graph.py --checkpoint .../step_005000 --video_id AAAA5

    # Specific .pt graph file:
    python infer_graph.py --checkpoint .../step_005000 --graph_file /path/to/graph.pt

    # Adjust resolution / steps:
    python infer_graph.py --checkpoint .../step_005000 --height 256 --width 256 --steps 50
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from torch_geometric.data import Batch

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model import TGNNEncoder
from adapter import GraphCrossAttnAdapter, pad_node_embeddings
from train import graph_to_prompt, encode_prompt

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

GRAPH_DIR   = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
CKPT_DIR    = "/projects/bgnv/leatherman/tgnn/checkpoints/step_005000"
HF_CACHE    = "/projects/bgnv/leatherman/hf_cache"
OUT_DIR     = "/projects/bgnv/leatherman/tgnn/outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_graph(graph_dir: str, video_id: str | None, split: str = "test"):
    """Load a graph .pt, picking randomly from the given split if no video_id."""
    graph_dir = Path(graph_dir)
    if video_id:
        pt = graph_dir / f"{video_id}.pt"
        if not pt.exists():
            sys.exit(f"Graph not found: {pt}")
        return torch.load(pt, weights_only=False)

    candidates = []
    for pt in graph_dir.glob("*.pt"):
        try:
            g = torch.load(pt, weights_only=False)
            if g.split == split:
                candidates.append(pt)
        except Exception:
            pass
    if not candidates:
        sys.exit(f"No graphs found for split={split} in {graph_dir}")
    pt = random.choice(candidates)
    print(f"  Selected graph: {pt.stem}  (split={split})")
    return torch.load(pt, weights_only=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}")

    # ── 1. Load pipeline ────────────────────────────────────────────────────
    print("\n[1] Loading CogVideoX-2b pipeline ...")
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # ── 2. Load checkpoint ──────────────────────────────────────────────────
    print(f"\n[2] Loading checkpoint: {args.checkpoint}")
    tgnn    = TGNNEncoder(d_model=256).to(device).eval()
    adapter = GraphCrossAttnAdapter(
        transformer    = pipe.transformer,
        d_graph        = 256,
        n_graph_tokens = 32,
        d_cross        = 256,
        n_heads        = 8,
    ).to(device).eval()

    tgnn.load_state_dict(torch.load(
        Path(args.checkpoint) / "tgnn.pt", map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(
        Path(args.checkpoint) / "adapter.pt", map_location=device, weights_only=True))
    print("  Checkpoint loaded.")

    # ── 3. Load graph ───────────────────────────────────────────────────────
    print("\n[3] Loading scene graph ...")
    if args.graph_file:
        graph = torch.load(args.graph_file, weights_only=False)
    else:
        graph = pick_graph(args.graph_dir, args.video_id, split=args.split)

    prompt = graph_to_prompt(graph)
    print(f"  Video ID : {graph.video_id}")
    print(f"  Nodes    : {graph.num_nodes}  Edges: {graph.num_edges}")
    print(f"  Prompt   : {prompt}")

    graph_batch = Batch.from_data_list([graph]).to(device)

    # ── 4. Encode graph → graph_emb ─────────────────────────────────────────
    print("\n[4] Encoding graph ...")
    with torch.no_grad():
        _, graph_emb = tgnn(graph_batch)   # [1, 256]
    print(f"  graph_emb: {graph_emb.shape}")

    # ── 5. Encode text → text_emb (positive + negative) ────────────────────
    print("\n[5] Encoding text ...")
    with torch.no_grad():
        text_emb = encode_prompt(
            [prompt], pipe.tokenizer, pipe.text_encoder,
            device=device, dtype=dtype,
        )  # [1, 226, 4096]

        neg_emb = encode_prompt(
            [""], pipe.tokenizer, pipe.text_encoder,
            device=device, dtype=dtype,
        )  # [1, 226, 4096] — unconditioned

    print(f"  text_emb : {text_emb.shape}")

    # ── 6. Cross-attention injection setup ──────────────────────────────────
    print("\n[6] Setting up cross-attention graph conditioning ...")
    with torch.no_grad():
        node_emb, _ = tgnn(graph_batch)
        graph_tokens = pad_node_embeddings(node_emb, graph_batch.batch, 32, 256).to(dtype)
    adapter.set_graph_tokens(graph_tokens)
    print(f"  graph_tokens : {graph_tokens.shape}  (hooks active on {len(adapter._hooks)} blocks)")

    # ── 7. Generate video ───────────────────────────────────────────────────
    print(f"\n[7] Generating video ({args.num_frames} frames, {args.steps} steps, "
          f"cfg={args.guidance_scale}, {args.height}×{args.width}) ...")

    # Hooks fire automatically inside pipe's transformer call.
    # prompt_embeds = text_emb (226 tokens, unchanged); graph injected via hooks.
    with torch.no_grad():
        output = pipe(
            prompt_embeds          = text_emb,
            negative_prompt_embeds = neg_emb,
            height                 = args.height,
            width                  = args.width,
            num_frames             = args.num_frames,
            num_inference_steps    = args.steps,
            guidance_scale         = args.guidance_scale,
            generator              = torch.Generator(device=device).manual_seed(args.seed),
        )
    adapter.clear_graph_tokens()

    frames = output.frames[0]  # list of PIL images

    # ── 8. Save ─────────────────────────────────────────────────────────────
    video_id = getattr(graph, "video_id", "output")
    out_path = Path(args.out_dir) / f"{video_id}_graph_cond.mp4"
    export_to_video(frames, str(out_path), fps=args.fps)
    print(f"\n[8] Saved → {out_path}")
    print(f"    Frames : {len(frames)}  FPS: {args.fps}")
    print(f"\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Graph-conditioned CogVideoX inference")
    p.add_argument("--checkpoint", default=CKPT_DIR)
    p.add_argument("--graph_dir",  default=GRAPH_DIR)
    p.add_argument("--graph_file", default=None,   help="Path to a specific .pt graph")
    p.add_argument("--video_id",   default=None,   help="Pick graph by video ID from graph_dir")
    p.add_argument("--split",      default="test", help="Graph split to sample from")
    p.add_argument("--hf_cache",   default=HF_CACHE)
    p.add_argument("--out_dir",    default=OUT_DIR)
    p.add_argument("--height",     type=int,   default=256)
    p.add_argument("--width",      type=int,   default=256)
    p.add_argument("--num_frames", type=int,   default=49)
    p.add_argument("--steps",      type=int,   default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--fps",        type=int,   default=8)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
