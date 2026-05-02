"""
eval_compositional.py
=====================
Compositional Fidelity Evaluation — the correct metric for this project.

Measures whether the generated video correctly depicts the objects specified
in the input scene graph, using CLIP zero-shot detection.

Method
------
For each test video:
  1. Extract N sampled frames from the generated video (.mp4)
  2. For each unique object in the scene graph (excluding person/background):
     - Compute CLIP similarity between each frame and "a photo of [object]"
     - Object is "detected" if max similarity across all frames > threshold
  3. Object Recall = detected_objects / total_graph_objects

Why this is the right metric
-----------------------------
CLIP Score (our previous metric) measured text-image alignment — useless here
because the text prompt is the same for both conditions.

Object Recall measures graph-image alignment: did the model generate a video
containing the objects the graph says should be there? This directly tests
whether graph conditioning adds visual information beyond the text.

Comparison
----------
  A: graph-conditioned (cross-attention, our model)
  B: text-only baseline (original CogVideoX, no graph adapter)
  Existing .mp4 files from eval_clip.py run are reused — no regeneration needed.

Usage
-----
    python eval_compositional.py --eval_dir .../outputs/eval_xattn
                                  --graph_dir .../dataset/ag/graphs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

sys.path.insert(0, os.path.dirname(__file__))

HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
EVAL_DIR  = "/projects/bgnv/leatherman/tgnn/outputs/eval_xattn_generic"

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
# Video → frames
# ---------------------------------------------------------------------------

def extract_frames(video_path: str, n_frames: int = 12) -> list[Image.Image]:
    """Sample n_frames uniformly from an mp4 and return as PIL images."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# CLIP zero-shot object recall
# ---------------------------------------------------------------------------

class ObjectRecallScorer:
    def __init__(self, device, threshold: float = 0.22):
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
    def _text_feat(self, label: str) -> torch.Tensor:
        txt = self.proc(
            text=[f"a photo of a {label} in an indoor scene"],
            return_tensors="pt", padding=True,
        ).to(self.device)
        out = self.model.text_model(
            input_ids=txt.input_ids, attention_mask=txt.attention_mask
        )
        feat = self.model.text_projection(out.pooler_output)
        return F.normalize(feat, dim=-1)   # [1, 512]

    @torch.no_grad()
    def _img_feats(self, frames: list[Image.Image]) -> torch.Tensor:
        """Batch encode all frames → [N, 512]."""
        all_feats = []
        bs = 8
        for i in range(0, len(frames), bs):
            batch = frames[i : i + bs]
            inp = self.proc(images=batch, return_tensors="pt").to(self.device)
            out = self.model.vision_model(pixel_values=inp.pixel_values)
            feat = self.model.visual_projection(out.pooler_output)
            all_feats.append(F.normalize(feat, dim=-1))
        return torch.cat(all_feats, dim=0)   # [N, 512]

    def recall(self, frames: list[Image.Image], object_names: list[str]) -> dict:
        """
        Returns dict with keys: recall, detected (list[bool]), scores (list[float])
        """
        if not frames or not object_names:
            return {"recall": 0.0, "detected": [], "scores": []}

        img_feats = self._img_feats(frames)   # [N, 512]

        detected = []
        scores   = []
        for obj in object_names:
            txt_feat = self._text_feat(obj)                  # [1, 512]
            sims = (img_feats @ txt_feat.T).squeeze(-1)      # [N]
            max_sim = float(sims.max())
            found = max_sim >= self.threshold
            detected.append(found)
            scores.append(round(max_sim, 4))

        recall = sum(detected) / len(detected)
        return {"recall": recall, "detected": detected, "scores": scores}


# ---------------------------------------------------------------------------
# Load graph objects
# ---------------------------------------------------------------------------

def get_graph_objects(graph_dir: str, video_id: str) -> list[str]:
    """Return unique non-person, non-background object names from the graph."""
    pt = Path(graph_dir) / f"{video_id}.pt"
    if not pt.exists():
        return []
    g = torch.load(pt, weights_only=False)
    unique_cls = g.node_class.unique().tolist()
    names = []
    for c in unique_cls:
        if 1 <= c < len(_OBJECT_NAMES):   # exclude 0=background
            names.append(_OBJECT_NAMES[c])
    return names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.environ["HF_HOME"] = args.hf_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    scorer = ObjectRecallScorer(device, threshold=args.threshold)

    # Find all video pairs (graph_cond + text_only)
    eval_dir = Path(args.eval_dir)
    graph_cond_videos = sorted(eval_dir.glob("*_graph_cond.mp4"))

    if not graph_cond_videos:
        sys.exit(f"No *_graph_cond.mp4 found in {eval_dir}")
    print(f"\nFound {len(graph_cond_videos)} video pairs in {eval_dir}\n")

    results = []

    for vid_path in graph_cond_videos:
        video_id   = vid_path.stem.replace("_graph_cond", "")
        text_path  = eval_dir / f"{video_id}_text_only.mp4"

        obj_names = get_graph_objects(args.graph_dir, video_id)
        if not obj_names:
            print(f"  [{video_id}] No graph found, skipping.")
            continue

        print(f"{'='*60}")
        print(f"  {video_id}  |  objects: {obj_names}")

        # Extract frames
        frames_a = extract_frames(str(vid_path),  n_frames=args.n_frames)
        frames_b = extract_frames(str(text_path), n_frames=args.n_frames) if text_path.exists() else []

        if not frames_a:
            print(f"  Could not decode {vid_path.name}, skipping.")
            continue

        # Compute recall
        res_a = scorer.recall(frames_a, obj_names)
        res_b = scorer.recall(frames_b, obj_names) if frames_b else {"recall": 0.0, "detected": [], "scores": []}

        delta = res_a["recall"] - res_b["recall"]
        print(f"  Recall [A] graph-cond : {res_a['recall']:.3f}  ({sum(res_a['detected'])}/{len(obj_names)} objects)")
        print(f"  Recall [B] text-only  : {res_b['recall']:.3f}  ({sum(res_b['detected'])}/{len(obj_names)} objects)")
        print(f"  Delta  (A - B)        : {delta:+.3f}")
        print(f"  Per-object scores [A]: {dict(zip(obj_names, res_a['scores']))}")

        results.append({
            "video_id":       video_id,
            "n_objects":      len(obj_names),
            "objects":        "|".join(obj_names),
            "recall_graph":   round(res_a["recall"], 4),
            "recall_text":    round(res_b["recall"], 4),
            "delta":          round(delta, 4),
            "detected_graph": sum(res_a["detected"]),
            "detected_text":  sum(res_b["detected"]),
        })

    if not results:
        print("No results computed.")
        return

    # Summary
    recalls_a = [r["recall_graph"] for r in results]
    recalls_b = [r["recall_text"]  for r in results]
    deltas    = [r["delta"]        for r in results]
    wins      = sum(d > 0 for d in deltas)

    print(f"\n{'='*60}")
    print("COMPOSITIONAL FIDELITY — SUMMARY")
    print(f"{'='*60}")
    print(f"  Mean Object Recall [A] graph-cond : {np.mean(recalls_a):.4f} ± {np.std(recalls_a):.4f}")
    print(f"  Mean Object Recall [B] text-only  : {np.mean(recalls_b):.4f} ± {np.std(recalls_b):.4f}")
    print(f"  Mean delta (A - B)                : {np.mean(deltas):+.4f} ± {np.std(deltas):.4f}")
    print(f"  Graph-cond wins                   : {wins}/{len(results)}")

    print(f"\n{'Video ID':<12} {'Recall_A':>9} {'Recall_B':>9} {'Delta':>7}  Objects")
    print("-" * 65)
    for r in results:
        print(f"  {r['video_id']:<10} {r['recall_graph']:>9.4f} {r['recall_text']:>9.4f} {r['delta']:>+7.4f}  {r['objects']}")

    csv_path = eval_dir / "eval_compositional.csv"
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
    p.add_argument("--eval_dir",   default=EVAL_DIR)
    p.add_argument("--graph_dir",  default=GRAPH_DIR)
    p.add_argument("--hf_cache",   default=HF_CACHE)
    p.add_argument("--n_frames",   type=int,   default=12)
    p.add_argument("--threshold",  type=float, default=0.22)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
