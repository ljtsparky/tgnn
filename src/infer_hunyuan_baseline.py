"""
infer_hunyuan_baseline.py
=========================
Pure HunyuanVideo text-only inference — NO graph adapter.
Baseline to assess HunyuanVideo quality before any fine-tuning.

Architecture notes
------------------
HunyuanVideo uses MMDiT (dual-stream → single-stream):
  - text_encoder  : LLaMA-3-8B  → sequence embeddings [B, seq, 4096]
  - text_encoder_2: CLIP         → pooled projection   [B, 768]
  - 20 × HunyuanVideoTransformerBlock     (double-stream, video + text)
  - 40 × HunyuanVideoSingleTransformerBlock (single-stream, merged)
  - d_inner = 24 heads × 128 = 3072
  - VAE: CausalVideoVAE, 4× temporal × 8× spatial compression

GPU notes
---------
Full pipeline ~36-40GB peak (with sequential CPU offload).
Use gpuH200x8 partition (80GB) for comfort; A100 (40GB) is very tight.

Usage
-----
    python infer_hunyuan_baseline.py --n_videos 3 --height 480 --width 848
"""

import argparse, os, random, sys
from pathlib import Path

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

sys.path.insert(0, os.path.dirname(__file__))

HF_CACHE    = "/projects/bgnv/leatherman/hf_cache"
GRAPH_DIR   = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
OUT_DIR     = "/projects/bgnv/leatherman/tgnn/outputs/hunyuan_baseline"
MODEL_ID    = "hunyuanvideo-community/HunyuanVideo"
PROMPT      = "a person in an indoor scene"


def pick_test_video_ids(graph_dir: str, n: int, seed: int) -> list[str]:
    """
    Same sampling logic as infer_hunyuan.list_test_graphs to ensure
    graph-cond and baseline runs cover the SAME 10 test videos.
    """
    candidates = []
    for pt in sorted(Path(graph_dir).glob("*.pt")):
        try:
            g = torch.load(pt, weights_only=False)
            if g.split == "test":
                candidates.append(g.video_id)
        except Exception:
            pass
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:n]


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}  |  Model: {MODEL_ID}")
    print(f"Resolution: {args.height}×{args.width}, Frames: {args.num_frames}")
    print(f"Prompt: '{PROMPT}'\n")

    # ── Load HunyuanVideo pipeline ──────────────────────────────────────────
    # transformer loaded in bf16; VAE needs fp16/fp32 for stability
    print("[1] Loading HunyuanVideo pipeline ...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        MODEL_ID, subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        MODEL_ID,
        transformer=transformer,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    # Sequential CPU offload: only active component stays on GPU
    # Reduces peak GPU memory from ~60GB to ~35-40GB
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_tiling()
    print("  Pipeline loaded with sequential CPU offload.\n")

    # ── Generate baseline videos ────────────────────────────────────────────
    video_ids = pick_test_video_ids(args.graph_dir, args.n_videos, args.seed)
    print(f"[2] Generating {len(video_ids)} baseline videos ...")

    for i, vid in enumerate(video_ids):
        print(f"\n  [{i+1}/{len(video_ids)}] {vid} ...")
        output = pipe(
            prompt              = PROMPT,
            height              = args.height,
            width               = args.width,
            num_frames          = args.num_frames,
            num_inference_steps = args.steps,
            guidance_scale      = args.guidance_scale,
            generator           = torch.Generator("cpu").manual_seed(args.seed + i),
        )
        # output.frames[0]: list of PIL images (num_frames frames)
        path = Path(args.out_dir) / f"{vid}_hunyuan_text_only_baseline.mp4"
        export_to_video(output.frames[0], str(path), fps=args.fps)
        print(f"  Saved → {path}")

    print(f"\n[3] Done. {len(video_ids)} videos in {args.out_dir}")
    print(f"  scp 'leatherman@dt-login.delta.ncsa.illinois.edu:{args.out_dir}/*.mp4' ./")


def parse_args():
    p = argparse.ArgumentParser(description="HunyuanVideo text-only baseline")
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--graph_dir",      default=GRAPH_DIR)
    p.add_argument("--hf_cache",       default=HF_CACHE)
    p.add_argument("--n_videos",       type=int,   default=3)
    p.add_argument("--height",         type=int,   default=480)
    p.add_argument("--width",          type=int,   default=848)
    p.add_argument("--num_frames",     type=int,   default=17)
    p.add_argument("--steps",          type=int,   default=30)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--fps",            type=int,   default=16)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
