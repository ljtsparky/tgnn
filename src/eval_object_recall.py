"""
eval_object_recall.py
=====================
Object Recall evaluation for graph-conditioned video generation.

Metric
------
For each generated video, given the ground-truth scene graph's set of
object classes (excluding person), compute whether CLIP zero-shot
classification detects each object in any frame of the video.

    Recall_per_video = |{detected GT objects}| / |{all GT objects}|
    Recall_overall   = mean over all videos

Comparison
----------
We run this for both:
  - graph-cond videos (outputs/hunyuan)
  - text-only baseline videos (outputs/hunyuan_baseline)

If our adapter actually injects object semantics, graph-cond should
score higher than baseline on the same scene graphs.

Detection method
----------------
Open-vocabulary classification with CLIP ViT-L/14:
  1. For each frame: compute image embedding
  2. For each of 35 object classes: compute text embedding "a photo of a {obj}"
  3. Cosine similarity → softmax → per-class probability
  4. Object is "detected" if max-frame probability > threshold (default 0.05)

Why softmax (not raw cosine):
  Raw cosine values are always low (~0.2-0.3) and not comparable across
  classes. Softmax over all 35 classes normalizes and gives interpretable
  probabilities — same approach as CLIP's zero-shot ImageNet eval.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import imageio.v3 as iio
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OBJECT_CLASSES_TXT = (
    "/projects/bgnv/leatherman/tgnn/dataset/ag/annotations/"
    "action_genome_v1.0/object_classes.txt"
)
GRAPH_DIR    = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
GRAPH_VIDEOS = "/projects/bgnv/leatherman/tgnn/outputs/hunyuan"
BASE_VIDEOS  = "/projects/bgnv/leatherman/tgnn/outputs/hunyuan_baseline"
HF_CACHE     = "/projects/bgnv/leatherman/hf_cache"

# AG compound class names → human-readable for CLIP
NAME_TO_PROMPT = {
    "person":         "person",
    "bag":            "bag",
    "bed":            "bed",
    "blanket":        "blanket",
    "book":           "book",
    "box":            "box",
    "broom":          "broom",
    "chair":          "chair",
    "closetcabinet":  "cabinet",
    "clothes":        "clothes",
    "cupglassbottle": "cup",
    "dish":           "dish",
    "door":           "door",
    "doorknob":       "doorknob",
    "doorway":        "doorway",
    "floor":          "floor",
    "food":           "food",
    "groceries":      "groceries",
    "laptop":         "laptop",
    "light":          "light",
    "medicine":       "medicine bottle",
    "mirror":         "mirror",
    "papernotebook":  "notebook",
    "phonecamera":    "phone",
    "picture":        "picture",
    "pillow":         "pillow",
    "refrigerator":   "refrigerator",
    "sandwich":       "sandwich",
    "shelf":          "shelf",
    "shoe":           "shoe",
    "sofacouch":      "sofa",
    "table":          "table",
    "television":     "television",
    "towel":          "towel",
    "vacuum":         "vacuum cleaner",
    "window":         "window",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_class_names() -> List[str]:
    """Returns list of 36 class names (idx 0=person, 1-35=objects)."""
    with open(OBJECT_CLASSES_TXT) as f:
        return [l.strip() for l in f if l.strip()]


def gt_objects(graph_path: Path, class_names: List[str]) -> List[int]:
    """Returns sorted list of unique non-person object class indices in a graph."""
    g = torch.load(graph_path, weights_only=False)
    classes = set(int(c) for c in g.node_class.tolist())
    classes.discard(0)   # remove person
    return sorted(classes)


def load_video_frames(mp4_path: Path) -> List[Image.Image]:
    """Load all frames from .mp4 as PIL Images (RGB)."""
    frames = iio.imread(str(mp4_path), plugin="pyav")  # [T, H, W, 3] uint8
    return [Image.fromarray(f) for f in frames]


# ---------------------------------------------------------------------------
# CLIP zero-shot detection
# ---------------------------------------------------------------------------

@torch.no_grad()
def detect_objects(
    frames: List[Image.Image],
    clip_model: CLIPModel,
    clip_proc: CLIPProcessor,
    text_embeds: torch.Tensor,    # [num_classes, dim]  pre-computed
    device: torch.device,
    threshold: float = 0.05,
) -> Dict[int, float]:
    """
    Returns: {class_idx: max_softmax_prob_over_frames}
    Object is considered "detected" if value >= threshold.
    """
    image_inputs = clip_proc(images=frames, return_tensors="pt").to(device)
    image_embeds = clip_model.get_image_features(**image_inputs)   # [T, dim]
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds  = F.normalize(text_embeds,  dim=-1)

    # cosine similarity scaled by CLIP's logit_scale
    logit_scale = clip_model.logit_scale.exp().item()
    sim = logit_scale * image_embeds @ text_embeds.T               # [T, C]
    probs = sim.softmax(dim=-1)                                    # [T, C]
    max_per_class = probs.max(dim=0).values.cpu().tolist()         # [C]
    return {i: p for i, p in enumerate(max_per_class)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    class_names = load_class_names()
    print(f"Loaded {len(class_names)} AG classes (idx 0=person, 1-{len(class_names)-1}=objects)")

    # CLIP setup
    print(f"\nLoading CLIP {args.clip_model} ...")
    clip_model = CLIPModel.from_pretrained(
        args.clip_model, cache_dir=HF_CACHE,
    ).to(device).eval()
    clip_proc  = CLIPProcessor.from_pretrained(args.clip_model, cache_dir=HF_CACHE)

    # Pre-compute text embeddings for all 35 object classes (skip person)
    object_names = class_names[1:]   # 35 entries
    prompts = [f"a photo of a {NAME_TO_PROMPT[n]}" for n in object_names]
    with torch.no_grad():
        tok = clip_proc(text=prompts, return_tensors="pt", padding=True).to(device)
        text_embeds = clip_model.get_text_features(**tok)   # [35, dim]
    print(f"  Text embeds: {text_embeds.shape}")

    # Find common video IDs across both folders
    graph_videos = {p.stem.replace("_hunyuan_graph_cond", ""): p
                    for p in Path(args.graph_dir).glob("*_graph_cond.mp4")}
    base_videos  = {p.stem.replace("_hunyuan_text_only_baseline", ""): p
                    for p in Path(args.baseline_dir).glob("*_baseline.mp4")}
    video_ids = sorted(set(graph_videos) & set(base_videos))
    print(f"\nFound {len(video_ids)} videos in both folders:")
    for v in video_ids:
        print(f"  - {v}")

    if not video_ids:
        print("\nNo overlapping videos yet. Run baseline inference first.")
        return

    # ─── Evaluation loop ───────────────────────────────────────────────────
    results = {"per_video": {}, "summary": {}}
    graph_pool, base_pool = [], []      # for overall recall

    for vid in video_ids:
        gt_idxs = gt_objects(Path(args.graph_dir_pt) / f"{vid}.pt", class_names)
        gt_names = [class_names[i] for i in gt_idxs]
        # gt_idxs are 1-indexed (skip person=0). text_embeds is 0-indexed for 35 obj.
        # → text_embeds row k corresponds to class k+1 (object_names[k] = class_names[k+1])
        gt_in_text = [i - 1 for i in gt_idxs]   # convert to text_embed row index

        if not gt_idxs:
            print(f"\n{vid}: no non-person objects in graph (skip)")
            continue

        print(f"\n=== {vid}: GT={gt_names} ({len(gt_idxs)} objects) ===")

        for tag, mp4_path, pool in [
            ("graph", graph_videos[vid], graph_pool),
            ("base",  base_videos[vid],  base_pool),
        ]:
            frames = load_video_frames(mp4_path)
            probs = detect_objects(frames, clip_model, clip_proc, text_embeds,
                                   device, threshold=args.threshold)
            detected = [k for k in gt_in_text if probs[k] >= args.threshold]
            recall = len(detected) / len(gt_in_text)
            pool.append(recall)
            detected_names = [object_names[k] for k in detected]
            print(f"  [{tag:5s}] recall={recall:.3f}  "
                  f"detected={detected_names}  "
                  f"top3={sorted(probs.items(), key=lambda x:-x[1])[:3]}")
            results["per_video"].setdefault(vid, {})[tag] = {
                "gt_objects":       gt_names,
                "detected_objects": detected_names,
                "recall":           recall,
            }

    if graph_pool and base_pool:
        results["summary"] = {
            "n_videos":        len(graph_pool),
            "graph_recall":    sum(graph_pool) / len(graph_pool),
            "baseline_recall": sum(base_pool)  / len(base_pool),
            "delta":           (sum(graph_pool) - sum(base_pool)) / len(graph_pool),
            "threshold":       args.threshold,
            "clip_model":      args.clip_model,
        }
        s = results["summary"]
        print(f"\n{'='*60}")
        print(f"  Object Recall Summary  (n={s['n_videos']}, threshold={s['threshold']})")
        print(f"{'='*60}")
        print(f"  Graph-cond recall : {s['graph_recall']:.4f}")
        print(f"  Baseline recall   : {s['baseline_recall']:.4f}")
        print(f"  Δ (improvement)   : {s['delta']:+.4f}")
        print(f"{'='*60}\n")

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph_dir",     default=GRAPH_VIDEOS,
                   help="folder with graph-cond .mp4 videos")
    p.add_argument("--baseline_dir",  default=BASE_VIDEOS,
                   help="folder with baseline .mp4 videos")
    p.add_argument("--graph_dir_pt",  default=GRAPH_DIR,
                   help="folder with graph .pt files (ground truth)")
    p.add_argument("--clip_model",    default="openai/clip-vit-large-patch14")
    p.add_argument("--threshold",     type=float, default=0.05,
                   help="softmax probability threshold for detection")
    p.add_argument("--out",           default="/projects/bgnv/leatherman/tgnn/outputs/object_recall.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
