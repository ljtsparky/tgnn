"""
eval_clip.py
============
Quantitative evaluation: CLIP score comparison between
  (A) Graph-conditioned generation  — TGNNEncoder + FiLM adapter
  (B) Text-only baseline            — original CogVideoX, no graph adapter

For each test video:
  1. Generate video A  (graph-conditioned)
  2. Generate video B  (text-only, same prompt + seed)
  3. Compute per-frame CLIP(frame, prompt) cosine similarity, average over frames
  4. Report delta = CLIP_A - CLIP_B

Output: eval_results.csv + printed table

Usage:
    python eval_clip.py --n_videos 10 --steps 50 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch_geometric.data import Batch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from transformers import CLIPProcessor, CLIPModel

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model import TGNNEncoder
from adapter import GraphCrossAttnAdapter, pad_node_embeddings
from train import graph_to_prompt, encode_prompt

GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
CKPT_DIR  = "/projects/bgnv/leatherman/tgnn/checkpoints_xattn_generic/step_005000"
HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
OUT_DIR   = "/projects/bgnv/leatherman/tgnn/outputs/eval_xattn_generic"


# ---------------------------------------------------------------------------
# CLIP scorer
# ---------------------------------------------------------------------------

class CLIPScorer:
    def __init__(self, device):
        print("  Loading CLIP (openai/clip-vit-base-patch32) ...")
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=HF_CACHE,
        ).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=HF_CACHE,
        )
        self.device = device

    @torch.no_grad()
    def score(self, frames: list[Image.Image], prompt: str) -> float:
        """Average CLIP cosine similarity between frames and prompt."""
        # Encode text via text_model + projection (avoids get_text_features API issues)
        txt = self.proc(text=[prompt], return_tensors="pt", padding=True).to(self.device)
        txt_out  = self.model.text_model(
            input_ids=txt.input_ids, attention_mask=txt.attention_mask
        )
        txt_feat = self.model.text_projection(txt_out.pooler_output)  # [1, 512]
        txt_feat = F.normalize(txt_feat, dim=-1)

        sims = []
        batch_size = 16
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i : i + batch_size]
            img = self.proc(images=batch_frames, return_tensors="pt").to(self.device)
            img_out  = self.model.vision_model(pixel_values=img.pixel_values)
            img_feat = self.model.visual_projection(img_out.pooler_output)  # [B, 512]
            img_feat = F.normalize(img_feat, dim=-1)
            sim = (img_feat @ txt_feat.T).squeeze(-1)                       # [B]
            sims.extend(sim.cpu().tolist())

        return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Graph loader
# ---------------------------------------------------------------------------

def load_test_graphs(graph_dir: str, n: int, seed: int) -> list:
    rng = __import__("random")
    rng.seed(seed)
    graph_dir = Path(graph_dir)
    candidates = []
    for pt in sorted(graph_dir.glob("*.pt")):
        try:
            g = torch.load(pt, weights_only=False)
            if g.split == "test":
                candidates.append(pt)
        except Exception:
            pass
    chosen = rng.sample(candidates, min(n, len(candidates)))
    graphs = [torch.load(p, weights_only=False) for p in chosen]
    print(f"  Loaded {len(graphs)} test graphs")
    return graphs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}\n")

    # ── Load pipeline ────────────────────────────────────────────────────────
    print("[1] Loading CogVideoX-2b pipeline ...")
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # ── Load graph-cond modules ──────────────────────────────────────────────
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
    adapter.to(dtype)   # match pipeline dtype (fp16)

    # ── Load CLIP scorer ─────────────────────────────────────────────────────
    print("\n[3] Loading CLIP scorer ...")
    scorer = CLIPScorer(device)

    # ── Load test graphs ─────────────────────────────────────────────────────
    print(f"\n[4] Sampling {args.n_videos} test graphs ...")
    graphs = load_test_graphs(args.graph_dir, args.n_videos, args.seed)

    # ── Evaluation loop ──────────────────────────────────────────────────────
    results = []
    gen_kwargs = dict(
        height=args.height, width=args.width,
        num_frames=args.num_frames, num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    for i, graph in enumerate(graphs):
        vid   = graph.video_id
        prompt = graph_to_prompt(graph)
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(graphs)}] {vid}  |  {prompt}")
        print(f"  Nodes: {graph.num_nodes}  Edges: {graph.num_edges}")

        gen  = torch.Generator(device=device).manual_seed(args.seed + i)

        # ── A: Graph-conditioned (cross-attn) ────────────────────────────
        print("  Generating [A] graph-conditioned ...")
        with torch.no_grad():
            graph_batch = Batch.from_data_list([graph]).to(device)
            node_emb, _ = tgnn(graph_batch)
            text_emb = encode_prompt(
                [prompt], pipe.tokenizer, pipe.text_encoder,
                device=device, dtype=dtype,
            )
            neg_emb = encode_prompt(
                [""], pipe.tokenizer, pipe.text_encoder,
                device=device, dtype=dtype,
            )
            graph_tokens = pad_node_embeddings(
                node_emb, graph_batch.batch, 32, 256
            ).to(dtype)
            adapter.set_graph_tokens(graph_tokens)

            out_a = pipe(
                prompt_embeds=text_emb, negative_prompt_embeds=neg_emb,
                generator=torch.Generator(device=device).manual_seed(args.seed + i),
                **gen_kwargs,
            )
            adapter.clear_graph_tokens()
        frames_a = out_a.frames[0]
        clip_a   = scorer.score(frames_a, prompt)

        path_a = Path(args.out_dir) / f"{vid}_graph_cond.mp4"
        export_to_video(frames_a, str(path_a), fps=args.fps)

        # ── B: Text-only baseline ─────────────────────────────────────────
        print("  Generating [B] text-only baseline ...")
        out_b = pipe(
            prompt=prompt, negative_prompt="",
            generator=torch.Generator(device=device).manual_seed(args.seed + i),
            **gen_kwargs,
        )
        frames_b = out_b.frames[0]
        clip_b   = scorer.score(frames_b, prompt)

        path_b = Path(args.out_dir) / f"{vid}_text_only.mp4"
        export_to_video(frames_b, str(path_b), fps=args.fps)

        delta = clip_a - clip_b
        print(f"  CLIP  [A] graph-cond : {clip_a:.4f}")
        print(f"  CLIP  [B] text-only  : {clip_b:.4f}")
        print(f"  Delta (A - B)        : {delta:+.4f}")

        results.append({
            "video_id": vid,
            "prompt":   prompt,
            "n_nodes":  graph.num_nodes,
            "n_edges":  graph.num_edges,
            "clip_graph_cond": round(clip_a, 4),
            "clip_text_only":  round(clip_b, 4),
            "delta":           round(delta, 4),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    clips_a = [r["clip_graph_cond"] for r in results]
    clips_b = [r["clip_text_only"]  for r in results]
    deltas  = [r["delta"]           for r in results]
    print(f"  Mean CLIP [A] graph-cond : {np.mean(clips_a):.4f} ± {np.std(clips_a):.4f}")
    print(f"  Mean CLIP [B] text-only  : {np.mean(clips_b):.4f} ± {np.std(clips_b):.4f}")
    print(f"  Mean delta (A - B)       : {np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")
    wins = sum(d > 0 for d in deltas)
    print(f"  Graph-cond wins          : {wins}/{len(results)}")

    print(f"\n{'Video ID':<12} {'CLIP_graph':>10} {'CLIP_text':>10} {'Delta':>8}")
    print("-" * 44)
    for r in results:
        print(f"  {r['video_id']:<10} {r['clip_graph_cond']:>10.4f} {r['clip_text_only']:>10.4f} {r['delta']:>+8.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = Path(args.out_dir) / "eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved → {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default=CKPT_DIR)
    p.add_argument("--graph_dir",      default=GRAPH_DIR)
    p.add_argument("--hf_cache",       default=HF_CACHE)
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--n_videos",       type=int,   default=10)
    p.add_argument("--height",         type=int,   default=256)
    p.add_argument("--width",          type=int,   default=256)
    p.add_argument("--num_frames",     type=int,   default=49)
    p.add_argument("--steps",          type=int,   default=50)
    p.add_argument("--guidance_scale", type=float, default=6.0)
    p.add_argument("--fps",            type=int,   default=8)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
