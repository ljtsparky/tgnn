"""
infer_wan_baseline.py
=====================
Pure Wan2.1-T2V-1.3B text-only inference — NO graph adapter, NO conditioning.
Used as the quality baseline to compare against graph-conditioned generation.

This tells us: what does Wan2.1 generate when given only the generic prompt
"a person in an indoor scene", with no graph information at all?

Usage
-----
    python infer_wan_baseline.py --n_videos 5 --seed 42
"""

import argparse, os, sys, random
from pathlib import Path

import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

# ── Paths ────────────────────────────────────────────────────────────────────
HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
OUT_DIR   = "/projects/bgnv/leatherman/tgnn/outputs/wan_baseline"

PROMPT    = "a person in an indoor scene"    # same generic prompt as graph-cond
NEG_PROMPT = ""


def pick_test_video_ids(graph_dir: str, n: int, seed: int) -> list[str]:
    """Sample n video IDs from the test split."""
    import torch as t
    rng = random.Random(seed)
    candidates = []
    for pt in sorted(Path(graph_dir).glob("*.pt")):
        try:
            g = t.load(pt, weights_only=False)
            if g.split == "test":
                candidates.append(g.video_id)
        except Exception:
            pass
    return rng.sample(candidates, min(n, len(candidates)))


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}")
    print(f"Prompt: '{PROMPT}'")
    print(f"Output: {args.out_dir}\n")

    # ── Load pure Wan2.1 pipeline (no modifications) ─────────────────────────
    print("[1] Loading Wan2.1-T2V-1.3B (text-only, no graph adapter) ...")
    # Try 14B first, fall back to 1.3B if not downloaded yet
    import os as _os
    cache = args.hf_cache + "/hub"
    model_id = ("Wan-AI/Wan2.1-T2V-14B-Diffusers"
                if _os.path.exists(f"{cache}/models--Wan-AI--Wan2.1-T2V-14B-Diffusers")
                else "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    print(f"  Model: {model_id}")
    pipe = WanPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    print("  Pipeline loaded.\n")

    # ── Pick test video IDs (to use same videos as graph-cond eval) ───────────
    video_ids = pick_test_video_ids(args.graph_dir, args.n_videos, args.seed)
    print(f"[2] Generating {len(video_ids)} baseline videos ...")

    for i, vid in enumerate(video_ids):
        print(f"\n  [{i+1}/{len(video_ids)}] {vid} ...")

        with torch.no_grad():
            output = pipe(
                prompt           = PROMPT,
                negative_prompt  = NEG_PROMPT,
                height           = args.height,
                width            = args.width,
                num_frames       = args.num_frames,
                num_inference_steps = args.steps,
                guidance_scale   = args.guidance_scale,
                generator        = torch.Generator(device=device).manual_seed(args.seed + i),
            )

        # Save: filename matches graph-cond format so they're easy to compare side-by-side
        path = Path(args.out_dir) / f"{vid}_wan_text_only_baseline.mp4"
        export_to_video(output.frames[0], str(path), fps=16)
        print(f"  Saved → {path}")

    print(f"\n[3] Done. {len(video_ids)} baseline videos saved to {args.out_dir}")
    print("Download and compare with graph-cond videos:")
    print(f"  scp 'leatherman@dt-login.delta.ncsa.illinois.edu:{args.out_dir}/*.mp4' ./")


def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 text-only baseline inference")
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--graph_dir",      default=GRAPH_DIR)
    p.add_argument("--hf_cache",       default=HF_CACHE)
    p.add_argument("--n_videos",       type=int,   default=5)
    p.add_argument("--height",         type=int,   default=480)
    p.add_argument("--width",          type=int,   default=832)
    p.add_argument("--num_frames",     type=int,   default=17)
    p.add_argument("--steps",          type=int,   default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
