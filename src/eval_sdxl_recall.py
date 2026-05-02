"""
eval_sdxl_recall.py
====================
Object Recall evaluation with GroundingDINO open-vocabulary detection.

For each test graph and its (graph_cond, baseline) image pair:
  1. Get GT object classes from graph (excluding person).
  2. Run GroundingDINO with prompts = the GT class names.
  3. Object detected if score >= threshold AND box area >= min_area.
  4. Per-image recall = |detected ∩ GT| / |GT|

Compared to CLIP zero-shot, GroundingDINO actually localizes objects
and gives confidence scores per box, much better fit for the task.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from transformers import (
    AutoProcessor, AutoModelForZeroShotObjectDetection,
)

CLASSES_TXT = (
    "/projects/bgnv/leatherman/tgnn/dataset/ag/annotations/"
    "action_genome_v1.0/object_classes.txt"
)
HF_CACHE  = "/projects/bgnv/leatherman/hf_cache"
GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"

NAME_TO_PROMPT = {
    "person":"person", "bag":"bag", "bed":"bed", "blanket":"blanket",
    "book":"book", "box":"box", "broom":"broom", "chair":"chair",
    "closetcabinet":"cabinet", "clothes":"clothes",
    "cupglassbottle":"cup", "dish":"dish", "door":"door",
    "doorknob":"doorknob", "doorway":"doorway", "floor":"floor",
    "food":"food", "groceries":"groceries", "laptop":"laptop",
    "light":"light", "medicine":"medicine bottle", "mirror":"mirror",
    "papernotebook":"notebook", "phonecamera":"phone",
    "picture":"picture", "pillow":"pillow", "refrigerator":"refrigerator",
    "sandwich":"sandwich", "shelf":"shelf", "shoe":"shoe",
    "sofacouch":"sofa", "table":"table", "television":"television",
    "towel":"towel", "vacuum":"vacuum cleaner", "window":"window",
}


def gt_objects(graph_path: Path, class_names: List[str]) -> List[str]:
    g = torch.load(graph_path, weights_only=False)
    classes = sorted(set(int(c) for c in g.node_class.tolist()) - {0})
    return [class_names[c] for c in classes]


def detect(image: Image.Image, prompt_objs: List[str], model, proc, device,
           threshold=0.30) -> List[str]:
    """Returns list of object names detected with score >= threshold."""
    text = ". ".join(NAME_TO_PROMPT[o] for o in prompt_objs) + "."
    inputs = proc(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    results = proc.post_process_grounded_object_detection(
        out, inputs.input_ids, threshold=threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    detected = set()
    for label in results["labels"]:
        for o in prompt_objs:
            if NAME_TO_PROMPT[o].lower() in label.lower():
                detected.add(o)
    return sorted(detected)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    class_names = [l.strip() for l in open(CLASSES_TXT) if l.strip()]

    print(f"\nLoading GroundingDINO {args.model_id} ...")
    proc  = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device).eval()

    img_dir = Path(args.img_dir)
    pairs = {}
    for p in img_dir.glob("*_graph_cond.png"):
        vid = p.stem.replace("_graph_cond", "")
        b = img_dir / f"{vid}_baseline.png"
        if b.exists():
            pairs[vid] = (p, b)
    print(f"Found {len(pairs)} (graph_cond, baseline) pairs")

    results = {"per_video": {}, "summary": {}}
    g_pool, b_pool = [], []

    for vid, (gp, bp) in sorted(pairs.items()):
        gt = gt_objects(Path(args.graph_dir) / f"{vid}.pt", class_names)
        if not gt:
            print(f"\n{vid}: no GT objects, skip"); continue

        det_g = detect(Image.open(gp).convert("RGB"), gt, model, proc, device,
                       threshold=args.threshold)
        det_b = detect(Image.open(bp).convert("RGB"), gt, model, proc, device,
                       threshold=args.threshold)
        rg = len(det_g) / len(gt); rb = len(det_b) / len(gt)
        g_pool.append(rg); b_pool.append(rb)
        print(f"\n{vid}: GT={gt}")
        print(f"  graph_cond: detected={det_g}  recall={rg:.3f}")
        print(f"  baseline  : detected={det_b}  recall={rb:.3f}")
        results["per_video"][vid] = {
            "gt": gt,
            "graph_detected":    det_g, "graph_recall":    rg,
            "baseline_detected": det_b, "baseline_recall": rb,
        }

    if g_pool:
        results["summary"] = {
            "n":            len(g_pool),
            "graph_recall":    sum(g_pool) / len(g_pool),
            "baseline_recall": sum(b_pool) / len(b_pool),
            "delta":           (sum(g_pool) - sum(b_pool)) / len(g_pool),
            "threshold":       args.threshold,
            "model":           args.model_id,
        }
        s = results["summary"]
        print(f"\n{'='*60}\n  Object Recall (GroundingDINO, threshold={s['threshold']})")
        print(f"  n = {s['n']}")
        print(f"  Graph-cond recall : {s['graph_recall']:.4f}")
        print(f"  Baseline recall   : {s['baseline_recall']:.4f}")
        print(f"  Δ                 : {s['delta']:+.4f}\n{'='*60}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nSaved → {args.out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir",     required=True, help="folder with _graph_cond.png and _baseline.png pairs")
    p.add_argument("--graph_dir",   default=GRAPH_DIR)
    p.add_argument("--model_id",    default="IDEA-Research/grounding-dino-tiny")
    p.add_argument("--threshold",   type=float, default=0.30)
    p.add_argument("--out",         default="/projects/bgnv/leatherman/tgnn/outputs/sdxl_recall.json")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
