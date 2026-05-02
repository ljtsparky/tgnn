"""
eval_wan.py
===========
Compositional Fidelity Evaluation for Wan2.1 + TGNNEncoder + WanGraphAdapter.

For each test graph:
  A) Graph-conditioned: adapter(node_tokens, text_emb) → cond → WanPipeline
  B) Text-only:  WanPipeline(prompt="a person in an indoor scene") — no adapter

Both conditions use the SAME generic prompt and SAME random seed.
Text-only does NOT contain object names → any object presence must come from
the model's indoor-scene prior, NOT from text conditioning.

Metric: Object Recall via CLIP zero-shot detection.
  For each graph object: max CLIP similarity over sampled frames > threshold
  Recall = detected_objects / total_graph_objects
"""

from __future__ import annotations

import argparse, csv, os, random, sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch_geometric.data import Batch
from diffusers import WanPipeline
from diffusers.utils import export_to_video
from transformers import CLIPProcessor, CLIPModel

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model  import TGNNEncoder
from adapter_wan import WanGraphAdapter
from adapter     import pad_node_embeddings
from train_wan   import encode_prompt_wan

GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
CKPT_DIR  = "/projects/bgnv/leatherman/tgnn/checkpoints_wan/step_005000"
HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
OUT_DIR   = "/projects/bgnv/leatherman/tgnn/outputs/eval_wan"

_OBJECT_NAMES = [
    "__background__",
    "bag", "bed", "blanket", "book", "box", "broom", "chair", "closet",
    "clothes", "bottle", "dish", "door", "doorknob", "doorway",
    "floor", "food", "groceries", "laptop", "light", "medicine",
    "mirror", "notebook", "phone", "picture", "pillow",
    "refrigerator", "sandwich", "shelf", "shoe", "sofa",
    "table", "television", "towel", "vacuum", "window",
]


# ---------------------------------------------------------------------------
# CLIP scorer
# ---------------------------------------------------------------------------

class ObjectRecallScorer:
    def __init__(self, device, threshold=0.22):
        print("  Loading CLIP (openai/clip-vit-base-patch32) ...")
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=HF_CACHE,
        ).to(device).eval()
        self.proc  = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", cache_dir=HF_CACHE,
        )
        self.device    = device
        self.threshold = threshold

    @torch.no_grad()
    def recall(self, frames: list[Image.Image], obj_names: list[str]) -> float:
        if not frames or not obj_names:
            return 0.0
        # Encode all frames at once
        inp = self.proc(images=frames, return_tensors="pt").to(self.device)
        img_out  = self.model.vision_model(pixel_values=inp.pixel_values)
        img_feat = F.normalize(self.model.visual_projection(img_out.pooler_output), dim=-1)

        detected = 0
        for obj in obj_names:
            txt = self.proc(
                text=[f"a photo of a {obj} in an indoor scene"],
                return_tensors="pt", padding=True,
            ).to(self.device)
            txt_out  = self.model.text_model(
                input_ids=txt.input_ids, attention_mask=txt.attention_mask,
            )
            txt_feat = F.normalize(
                self.model.text_projection(txt_out.pooler_output), dim=-1,
            )
            max_sim = float((img_feat @ txt_feat.T).max())
            if max_sim >= self.threshold:
                detected += 1
        return detected / len(obj_names)


def extract_frames(path: str, n: int = 5) -> list[Image.Image]:
    """Sample n frames uniformly from mp4."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release(); return []
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def get_graph_objects(graph) -> list[str]:
    unique_cls = graph.node_class.unique().tolist()
    return [_OBJECT_NAMES[c] for c in unique_cls if 1 <= c < len(_OBJECT_NAMES)]


def load_test_graphs(graph_dir, n, seed):
    rng = random.Random(seed)
    cands = []
    for pt in sorted(Path(graph_dir).glob("*.pt")):
        try:
            g = torch.load(pt, weights_only=False)
            if g.split == "test":
                cands.append(pt)
        except Exception:
            pass
    chosen = rng.sample(cands, min(n, len(cands)))
    return [torch.load(p, weights_only=False) for p in chosen]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}  dtype: {dtype}\n")

    # ── Load pipeline ────────────────────────────────────────────────────────
    print("[1] Loading Wan2.1 pipeline ...")
    pipe = WanPipeline.from_pretrained(
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=dtype, local_files_only=True,
    ).to(device)
    pipe.vae.enable_slicing(); pipe.vae.enable_tiling()

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print(f"\n[2] Loading checkpoint: {args.checkpoint}")
    d_text  = pipe.transformer.config.text_dim
    tgnn    = TGNNEncoder(d_model=256).to(device).eval()
    adapter = WanGraphAdapter(d_graph=256, d_text=d_text, n_graph_tokens=32).to(device=device, dtype=dtype).eval()
    tgnn.load_state_dict(torch.load(
        Path(args.checkpoint) / "tgnn.pt", map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(
        Path(args.checkpoint) / "adapter.pt", map_location=device, weights_only=True))

    # ── CLIP scorer ──────────────────────────────────────────────────────────
    print("\n[3] Loading CLIP scorer ...")
    scorer = ObjectRecallScorer(device, threshold=args.threshold)

    # ── Test graphs ──────────────────────────────────────────────────────────
    print(f"\n[4] Sampling {args.n_videos} test graphs ...")
    graphs = load_test_graphs(args.graph_dir, args.n_videos, args.seed)

    gen_kwargs = dict(
        height=args.height, width=args.width,
        num_frames=args.num_frames, num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )

    results = []
    for i, graph in enumerate(graphs):
        vid      = graph.video_id
        obj_names = get_graph_objects(graph)
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(graphs)}] {vid}  |  objects: {obj_names}")

        # ── A: Graph-conditioned ──────────────────────────────────────────
        print("  Generating [A] graph-conditioned ...")
        with torch.no_grad():
            graph_batch = Batch.from_data_list([graph]).to(device)
            node_emb, _ = tgnn(graph_batch)
            node_tokens = pad_node_embeddings(node_emb, graph_batch.batch, 32, 256).to(dtype)
            text_emb    = encode_prompt_wan(
                ["a person in an indoor scene"], pipe.tokenizer, pipe.text_encoder,
                device=device, dtype=dtype,
            )
            neg_emb = encode_prompt_wan(
                [""], pipe.tokenizer, pipe.text_encoder,
                device=device, dtype=dtype,
            )
            cond     = adapter(node_tokens, text_emb)
            neg_cond = adapter(torch.zeros_like(node_tokens), neg_emb)

            out_a = pipe(
                prompt_embeds=cond, negative_prompt_embeds=neg_cond,
                generator=torch.Generator(device=device).manual_seed(args.seed + i),
                **gen_kwargs,
            )
        path_a = Path(args.out_dir) / f"{vid}_graph_cond.mp4"
        export_to_video(out_a.frames[0], str(path_a), fps=16)
        frames_a = extract_frames(str(path_a), n=args.n_eval_frames)
        recall_a = scorer.recall(frames_a, obj_names)

        # ── B: Text-only baseline ─────────────────────────────────────────
        print("  Generating [B] text-only baseline ...")
        out_b = pipe(
            prompt="a person in an indoor scene", negative_prompt="",
            generator=torch.Generator(device=device).manual_seed(args.seed + i),
            **gen_kwargs,
        )
        path_b = Path(args.out_dir) / f"{vid}_text_only.mp4"
        export_to_video(out_b.frames[0], str(path_b), fps=16)
        frames_b = extract_frames(str(path_b), n=args.n_eval_frames)
        recall_b = scorer.recall(frames_b, obj_names)

        delta = recall_a - recall_b
        print(f"  Object Recall [A] graph-cond : {recall_a:.3f}")
        print(f"  Object Recall [B] text-only  : {recall_b:.3f}")
        print(f"  Delta (A - B)                : {delta:+.3f}")
        print(f"  Objects: {obj_names}")

        results.append({
            "video_id":     vid,
            "objects":      "|".join(obj_names),
            "n_objects":    len(obj_names),
            "recall_graph": round(recall_a, 4),
            "recall_text":  round(recall_b, 4),
            "delta":        round(delta, 4),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    ra = [r["recall_graph"] for r in results]
    rb = [r["recall_text"]  for r in results]
    d  = [r["delta"]        for r in results]
    wins = sum(x > 0 for x in d)

    print(f"\n{'='*60}")
    print("OBJECT RECALL — SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean Recall [A] graph-cond : {np.mean(ra):.4f} ± {np.std(ra):.4f}")
    print(f"  Mean Recall [B] text-only  : {np.mean(rb):.4f} ± {np.std(rb):.4f}")
    print(f"  Mean delta (A - B)         : {np.mean(d):+.4f} ± {np.std(d):.4f}")
    print(f"  Graph-cond wins            : {wins}/{len(results)}")

    print(f"\n{'Video':<10} {'Recall_A':>9} {'Recall_B':>9} {'Delta':>7}  Objects")
    print("-" * 65)
    for r in results:
        print(f"  {r['video_id']:<8} {r['recall_graph']:>9.4f} {r['recall_text']:>9.4f} {r['delta']:>+7.4f}  {r['objects']}")

    csv_path = Path(args.out_dir) / "eval_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults → {csv_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     default=CKPT_DIR)
    p.add_argument("--graph_dir",      default=GRAPH_DIR)
    p.add_argument("--hf_cache",       default=HF_CACHE)
    p.add_argument("--out_dir",        default=OUT_DIR)
    p.add_argument("--n_videos",       type=int,   default=10)
    p.add_argument("--height",         type=int,   default=480)
    p.add_argument("--width",          type=int,   default=832)
    p.add_argument("--num_frames",     type=int,   default=17)
    p.add_argument("--steps",          type=int,   default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--n_eval_frames",  type=int,   default=5)
    p.add_argument("--threshold",      type=float, default=0.22)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
