"""
inspect_graphs.py
==================
Sampling and quality-check tool for the .pt graph files produced by ag_graph_dataset.py.

Usage:
    # Quick check on 5 random graphs
    python inspect_graphs.py --graph_dir /path/to/graphs

    # Check specific video
    python inspect_graphs.py --graph_dir /path/to/graphs --video_id 0A8CF

    # Dataset-wide statistics (slower, scans all .pt files)
    python inspect_graphs.py --graph_dir /path/to/graphs --stats

    # Show edge details for one graph
    python inspect_graphs.py --graph_dir /path/to/graphs --video_id 0A8CF --verbose
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Class name lists  (must match ag_graph_dataset.py)
# ---------------------------------------------------------------------------

OBJECT_CLASSES = [
    "__background__",           # index 0 in file → skip
    "person",                   # PERSON_CLASS_IDX = 0 in our node_class
    # object_classes.txt indices 1-34 → node_class 2-35 after OBJECT_CLASS_OFFSET
]

# Inline fallback lists so this script runs standalone without the .txt files
_OBJECT_NAMES_RAW = [
    "__background__",
    "bag", "bed", "blanket", "book", "box", "broom", "chair", "closet/cabinet",
    "clothes", "cup/glass/bottle", "dish", "door", "doorknob", "doorway",
    "floor", "food", "groceries", "laptop", "light", "medicine",
    "mirror", "paper/notebook", "phone/camera", "picture", "pillow",
    "refrigerator", "sandwich", "shelf", "shoe", "sofa/couch",
    "table", "television", "towel", "vacuum", "window",
]

ATTENTION_CLASSES   = ["looking_at", "not_looking_at", "unsure"]
SPATIAL_CLASSES     = ["in_front_of", "behind", "above", "beneath", "in", "on_the_side_of"]
CONTACTING_CLASSES  = [
    "holding", "not_contacting", "touching", "sitting_on", "leaning_on",
    "other_relationship", "standing_on", "wearing", "lying_on", "covered_by",
    "carrying", "drinking_from", "eating", "writing_on", "wiping",
    "have_it_on_the_back", "twisting",
]

EDGE_ATTR_DIM = 28

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def node_class_name(cls_idx: int) -> str:
    if cls_idx == 0:
        return "person"
    # node_class = line number in object_classes.txt (1-35)
    # _OBJECT_NAMES_RAW index 0 = "__background__" (line 0, skipped)
    # so node_class=1 → _OBJECT_NAMES_RAW[1] = "bag", no offset needed
    if cls_idx < len(_OBJECT_NAMES_RAW):
        return _OBJECT_NAMES_RAW[cls_idx]
    return f"obj_{cls_idx}"


def decode_edge_attr(attr: torch.Tensor):
    """Return human-readable dict from a single edge_attr row."""
    a = attr.numpy()
    attn  = [ATTENTION_CLASSES[i]  for i in range(3)  if a[i]     > 0.5]
    spat  = [SPATIAL_CLASSES[i]    for i in range(6)  if a[3+i]   > 0.5]
    cont  = [CONTACTING_CLASSES[i] for i in range(17) if a[9+i]   > 0.5]
    iou   = float(a[26])
    gap   = float(a[27])
    return dict(attention=attn, spatial=spat, contacting=cont, iou=iou, gap_norm=gap)


# ---------------------------------------------------------------------------
# Single-graph printer
# ---------------------------------------------------------------------------

def print_graph(data, verbose: bool = False) -> None:
    vid   = data.video_id
    T     = data.num_frames
    N     = data.node_class.shape[0]
    E     = data.edge_index.shape[1] if data.edge_index.numel() > 0 else 0

    n_spatial  = int((data.edge_type == 0).sum())
    n_temporal = int((data.edge_type == 1).sum())

    frame_ids = data.frame_ids.tolist()
    gaps = [frame_ids[i+1] - frame_ids[i] for i in range(len(frame_ids)-1)]
    avg_gap = float(np.mean(gaps)) if gaps else 0.0

    # Nodes per frame
    nodes_per_frame = []
    for t in range(T):
        nodes_per_frame.append(int((data.node_frame == t).sum()))

    # Temporal edge IoU distribution
    if n_temporal > 0:
        temp_mask = data.edge_type == 1
        temp_ious = data.edge_attr[temp_mask, 26]
        iou_mean  = float(temp_ious.mean())
        iou_min   = float(temp_ious.min())
        low_conf  = int((temp_ious < 0.3).sum())
    else:
        iou_mean = iou_min = 0.0
        low_conf = 0

    print(f"\n{'='*60}")
    print(f"  Video : {vid}   split={data.split}")
    print(f"{'='*60}")
    print(f"  Annotated frames : {T}   frame IDs: {frame_ids[:5]}{'…' if T>5 else ''}")
    print(f"  Avg frame gap    : {avg_gap:.1f} frames")
    print(f"  Total nodes      : {N}   (avg {N/T:.1f} per frame)")
    print(f"  Nodes per frame  : {nodes_per_frame[:10]}{'…' if T>10 else ''}")
    print(f"  Total edges      : {E}")
    print(f"    ├─ spatial     : {n_spatial}")
    print(f"    └─ temporal    : {n_temporal}")
    if n_temporal > 0:
        print(f"  Temporal IoU     : mean={iou_mean:.3f}  min={iou_min:.3f}  "
              f"low-conf(<0.3): {low_conf}/{n_temporal}")
    print(f"  node_class range : [{int(data.node_class.min())}, {int(data.node_class.max())}]")
    print(f"  node_bbox range  : x∈[{float(data.node_bbox[:,0].min()):.2f}, "
          f"{float(data.node_bbox[:,0].max()):.2f}]  "
          f"y∈[{float(data.node_bbox[:,1].min()):.2f}, "
          f"{float(data.node_bbox[:,1].max()):.2f}]")

    if verbose:
        print(f"\n  --- Node list (first 20) ---")
        for i in range(min(20, N)):
            cls  = int(data.node_class[i])
            frm  = int(data.node_frame[i])
            bbox = data.node_bbox[i].tolist()
            print(f"    node {i:3d}  frame={frm}  class={cls:2d} ({node_class_name(cls):30s})"
                  f"  bbox=[{bbox[0]:.2f},{bbox[1]:.2f},{bbox[2]:.2f},{bbox[3]:.2f}]")

        print(f"\n  --- Spatial edges (first 10) ---")
        sp_mask = (data.edge_type == 0).nonzero(as_tuple=True)[0]
        for idx in sp_mask[:10]:
            src = int(data.edge_index[0, idx])
            dst = int(data.edge_index[1, idx])
            dec = decode_edge_attr(data.edge_attr[idx])
            print(f"    {node_class_name(int(data.node_class[src]))} → "
                  f"{node_class_name(int(data.node_class[dst]))} | "
                  f"attn={dec['attention']} spat={dec['spatial']} cont={dec['contacting']}")

        print(f"\n  --- Temporal edges (first 10) ---")
        te_mask = (data.edge_type == 1).nonzero(as_tuple=True)[0]
        for idx in te_mask[:10]:
            src = int(data.edge_index[0, idx])
            dst = int(data.edge_index[1, idx])
            dec = decode_edge_attr(data.edge_attr[idx])
            src_frm = int(data.node_frame[src])
            dst_frm = int(data.node_frame[dst])
            print(f"    frame {src_frm}→{dst_frm}  "
                  f"{node_class_name(int(data.node_class[src]))} → "
                  f"{node_class_name(int(data.node_class[dst]))}  "
                  f"IoU={dec['iou']:.3f}  gap_norm={dec['gap_norm']:.3f}")


# ---------------------------------------------------------------------------
# Dataset-wide statistics
# ---------------------------------------------------------------------------

def print_dataset_stats(graph_dir: str, max_files: Optional[int] = None) -> None:
    pt_files = sorted(Path(graph_dir).glob("*.pt"))
    if max_files:
        pt_files = pt_files[:max_files]

    print(f"\nScanning {len(pt_files)} graph files in {graph_dir} …\n")

    n_frames_list     = []
    n_nodes_list      = []
    n_spatial_list    = []
    n_temporal_list   = []
    avg_gap_list      = []
    iou_all           = []
    n_train = n_test  = 0

    for pt in pt_files:
        try:
            data = torch.load(pt, weights_only=False)
        except Exception as e:
            print(f"  [WARN] Could not load {pt.name}: {e}")
            continue

        T  = data.num_frames
        N  = data.node_class.shape[0]
        n_spatial  = int((data.edge_type == 0).sum())
        n_temporal = int((data.edge_type == 1).sum())

        n_frames_list.append(T)
        n_nodes_list.append(N)
        n_spatial_list.append(n_spatial)
        n_temporal_list.append(n_temporal)

        frame_ids = data.frame_ids.tolist()
        gaps = [frame_ids[i+1] - frame_ids[i] for i in range(len(frame_ids)-1)]
        if gaps:
            avg_gap_list.append(np.mean(gaps))

        if n_temporal > 0:
            te_mask = data.edge_type == 1
            iou_all.extend(data.edge_attr[te_mask, 26].tolist())

        if data.split == "train":
            n_train += 1
        else:
            n_test += 1

    def summary(lst, name):
        arr = np.array(lst)
        print(f"  {name:30s}  mean={arr.mean():.1f}  "
              f"min={arr.min():.0f}  max={arr.max():.0f}  "
              f"p25={np.percentile(arr,25):.0f}  p75={np.percentile(arr,75):.0f}")

    total = len(n_frames_list)
    print(f"{'='*60}")
    print(f"  Dataset-wide Statistics  ({total} videos)")
    print(f"{'='*60}")
    print(f"  Train / Test split : {n_train} / {n_test}")
    summary(n_frames_list,   "annotated frames per video")
    summary(n_nodes_list,    "nodes per video")
    summary(n_spatial_list,  "spatial edges per video")
    summary(n_temporal_list, "temporal edges per video")
    summary(avg_gap_list,    "avg frame gap per video")

    if iou_all:
        iou_arr = np.array(iou_all)
        print(f"\n  Temporal edge IoU distribution ({len(iou_arr)} edges total):")
        print(f"    mean={iou_arr.mean():.3f}  median={np.median(iou_arr):.3f}  "
              f"min={iou_arr.min():.3f}  max={iou_arr.max():.3f}")
        for thresh in [0.1, 0.3, 0.5, 0.7]:
            pct = 100 * (iou_arr >= thresh).mean()
            print(f"    IoU ≥ {thresh:.1f} : {pct:.1f}% of temporal edges")

    # Sanity checks
    print(f"\n  Sanity checks:")
    zero_temporal = sum(1 for n in n_temporal_list if n == 0)
    print(f"    Videos with 0 temporal edges : {zero_temporal} / {total}")
    if iou_all:
        low_iou = sum(1 for v in iou_all if v < IOU_THRESHOLD_DISPLAY)
        print(f"    Temporal edges with IoU < 0.3 : {low_iou} / {len(iou_all)} "
              f"({100*low_iou/len(iou_all):.1f}%)")

IOU_THRESHOLD_DISPLAY = 0.3


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect Action Genome graph .pt files")
    parser.add_argument(
        "--graph_dir",
        default="/projects/bgnv/leatherman/tgnn/dataset/ag/graphs",
        help="Directory containing .pt graph files",
    )
    parser.add_argument("--video_id",  default=None, help="Inspect a specific video ID (e.g. 0A8CF)")
    parser.add_argument("--n_sample",  type=int, default=5,   help="Number of random graphs to inspect")
    parser.add_argument("--stats",     action="store_true",   help="Print dataset-wide statistics")
    parser.add_argument("--max_files", type=int, default=None,help="Limit files scanned for --stats")
    parser.add_argument("--verbose",   action="store_true",   help="Print node and edge details")
    args = parser.parse_args()

    graph_dir = args.graph_dir

    if not os.path.isdir(graph_dir):
        print(f"[ERROR] graph_dir not found: {graph_dir}")
        print("  Have you run ag_graph_dataset.py yet?")
        return

    pt_files = list(Path(graph_dir).glob("*.pt"))
    if len(pt_files) == 0:
        print(f"[ERROR] No .pt files found in {graph_dir}")
        return

    print(f"Found {len(pt_files)} .pt files in {graph_dir}")

    # --- Specific video ---
    if args.video_id:
        pt_path = os.path.join(graph_dir, f"{args.video_id}.pt")
        if not os.path.exists(pt_path):
            print(f"[ERROR] {pt_path} not found")
            return
        data = torch.load(pt_path, weights_only=False)
        print_graph(data, verbose=args.verbose)
        return

    # --- Dataset-wide stats ---
    if args.stats:
        print_dataset_stats(graph_dir, max_files=args.max_files)
        return

    # --- Random sample ---
    sample = random.sample(pt_files, min(args.n_sample, len(pt_files)))
    for pt in sample:
        try:
            data = torch.load(pt, weights_only=False)
            print_graph(data, verbose=args.verbose)
        except Exception as e:
            print(f"[WARN] Could not load {pt.name}: {e}")


if __name__ == "__main__":
    main()