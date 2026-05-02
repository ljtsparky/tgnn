"""
build_pseudo_sg.py
==================
Build pseudo scene graphs for large-scale video datasets (Panda-70M, WebVid, etc.)
to enable mixed training: backbone data (no graph) + AG data (real graph).

Strategy
--------
For each video clip:
1. Extract keyframes (e.g., 1 frame per second)
2. Run object detector (YOLOv8 or DINO) on each frame
3. Detect person bbox
4. Build spatial relations from relative bboxes (above/below/left/right/overlap)
5. Build temporal edges across frames (same object class with IoU > threshold)
6. Save as PyG Data in same format as Action Genome graphs

Why pseudo graphs are academically valid
-----------------------------------------
The scientific contribution is the CONDITIONING MECHANISM (T-GNN + adapter).
The scene graph is the INPUT signal, not the output. Using detector-generated
pseudo graphs as input conditioning is analogous to ControlNet using Canny edges
from the training image itself — it's a form of self-supervised conditioning.

The academic novelty is preserved: at inference time, the user provides a DESIRED
scene graph (specifying the intended object relationships), and the model generates
video consistent with it. The pseudo-label quality determines training signal
quality, not the contribution's validity.

Object classes used (COCO → AG mapping)
-----------------------------------------
COCO detections are mapped to Action Genome classes where possible.
For backbone data, we only need approximate graph structure, not perfect alignment.

Usage
-----
    # Process a directory of video clips:
    python build_pseudo_sg.py \
        --video_dir /data/panda70m/clips \
        --out_dir   /data/panda70m/graphs \
        --n_workers 8

    # Process WebVid:
    python build_pseudo_sg.py \
        --video_dir /data/webvid/videos \
        --out_dir   /data/webvid/graphs

Requirements
------------
    pip install ultralytics  # YOLOv8
"""

from __future__ import annotations

import argparse, os, sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch_geometric.data import Data

# COCO class ID → Action Genome class ID mapping (best-effort)
# COCO: https://cocodataset.org/#explore  (80 classes)
# AG:   object_classes.txt                (36 classes)
COCO_TO_AG = {
    0:  None,   # person → we handle separately
    24: 1,      # backpack → bag
    26: 1,      # handbag → bag
    57: 2,      # couch → sofa (not exact, but closest)
    59: 2,      # bed → bed
    56: 7,      # chair → chair
    73: 4,      # book → book
    39: 10,     # bottle → cup/glass/bottle
    41: 10,     # cup → cup/glass/bottle
    45: 11,     # bowl → dish
    63: 18,     # laptop → laptop
    67: 23,     # cell phone → phone/camera
    62: 32,     # tv → television
    72: 26,     # refrigerator → refrigerator
    48: 27,     # sandwich → sandwich
    60: 31,     # dining table → table
    28: 25,     # umbrella → (pillow as fallback)
    74: 33,     # clock → (towel as fallback, imperfect)
}

# Spatial relation encoding (6-dim one-hot: above, below, left, right, overlap, contains)
def spatial_relation(p_bbox, o_bbox):
    """
    p_bbox, o_bbox: [x1, y1, x2, y2] normalized
    Returns 6-dim spatial encoding
    """
    px1, py1, px2, py2 = p_bbox
    ox1, oy1, ox2, oy2 = o_bbox
    pc = ((px1+px2)/2, (py1+py2)/2)
    oc = ((ox1+ox2)/2, (oy1+oy2)/2)

    # Compute IoU
    ix1 = max(px1, ox1); iy1 = max(py1, oy1)
    ix2 = min(px2, ox2); iy2 = min(py2, oy2)
    iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
    iou = (iw*ih) / max((px2-px1)*(py2-py1)+(ox2-ox1)*(oy2-oy1)-iw*ih, 1e-6)

    vec = np.zeros(6, dtype=np.float32)
    if iou > 0.5:
        vec[4] = 1.0   # overlap
    elif pc[1] < oc[1] - 0.1:
        vec[0] = 1.0   # above
    elif pc[1] > oc[1] + 0.1:
        vec[1] = 1.0   # below
    elif pc[0] < oc[0] - 0.1:
        vec[2] = 1.0   # left
    else:
        vec[3] = 1.0   # right
    vec[5] = iou
    return vec


def build_graph_from_detections(
    frame_detections: List[dict],   # per-frame: {boxes, labels, confs, frame_h, frame_w}
    video_id: str,
    split: str = "train",
    iou_threshold: float = 0.3,
) -> Data | None:
    """
    Build a PyG Data object in Action Genome format from YOLO detections.
    Returns None if no person detected in any frame.
    """
    node_classes = []
    node_bboxes  = []
    node_frames  = []
    frame_ids    = []

    # Build nodes
    for f_idx, det in enumerate(frame_detections):
        boxes  = det["boxes"]     # [[x1,y1,x2,y2], ...]
        labels = det["labels"]    # list of COCO class IDs
        H, W   = det["frame_h"], det["frame_w"]
        frame_ids.append(det.get("frame_id", f_idx * 10))

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            # Normalize
            xc = (x1 + x2) / 2 / W
            yc = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H

            if label == 0:  # person
                ag_class = 0
            else:
                ag_class = COCO_TO_AG.get(int(label), None)
                if ag_class is None:
                    continue  # skip unmapped classes

            node_classes.append(ag_class)
            node_bboxes.append([xc, yc, bw, bh])
            node_frames.append(f_idx)

    if not node_classes or 0 not in node_classes:
        return None  # no person → skip

    N = len(node_classes)
    node_class = torch.tensor(node_classes, dtype=torch.long)
    node_bbox  = torch.tensor(node_bboxes,  dtype=torch.float32)
    node_frame = torch.tensor(node_frames,  dtype=torch.long)

    # Build edges (spatial within frame, temporal across frames)
    edge_index = []
    edge_attr  = []
    edge_type  = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            fi, fj = node_frames[i], node_frames[j]
            ci, cj = node_classes[i], node_classes[j]

            if fi == fj and ci == 0:  # spatial edge: person → object in same frame
                if cj != 0:
                    sp = spatial_relation(node_bboxes[i], node_bboxes[j])
                    # edge_attr: 4 attn (zeros) + 6 spatial + 17 contacting (zeros) + 1 iou
                    attr = np.zeros(28, dtype=np.float32)
                    attr[4:10] = sp[:6]
                    attr[27]   = sp[5]  # IoU
                    edge_index.append([i, j])
                    edge_attr.append(attr)
                    edge_type.append(0)

            elif fi != fj and ci == cj:  # temporal: same class across frames
                # IoU between bboxes (using center distance as proxy)
                b1, b2 = node_bboxes[i], node_bboxes[j]
                dx = abs(b1[0] - b2[0]); dy = abs(b1[1] - b2[1])
                if dx < 0.3 and dy < 0.3:  # rough proximity check
                    gap = abs(fi - fj)
                    attr = np.zeros(28, dtype=np.float32)
                    attr[27] = max(0, 1.0 - (dx + dy))  # proximity as pseudo-IoU
                    edge_index.append([i, j])
                    edge_attr.append(attr)
                    edge_type.append(1)

    if not edge_index:
        return None

    data = Data(
        node_class  = node_class,
        node_bbox   = node_bbox,
        node_frame  = node_frame,
        edge_index  = torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr   = torch.tensor(np.stack(edge_attr), dtype=torch.float32),
        edge_type   = torch.tensor(edge_type, dtype=torch.long),
        frame_ids   = torch.tensor(frame_ids, dtype=torch.long),
        video_id    = video_id,
        split       = split,
        pseudo      = True,   # mark as pseudo-annotated
    )
    return data


def process_video(video_path: str, model, out_dir: str, split: str, fps_sample: float = 1.0):
    """Extract frames, run detection, build graph, save."""
    video_id = Path(video_path).stem
    out_path = Path(out_dir) / f"{video_id}.pt"
    if out_path.exists():
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_interval = max(1, int(fps / fps_sample))
    frame_detections = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            H, W = frame.shape[:2]
            results = model(frame, verbose=False)[0]
            boxes  = results.boxes.xyxy.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy().astype(int)
            frame_detections.append({
                "boxes":   boxes.tolist() if len(boxes) else [],
                "labels":  labels.tolist() if len(labels) else [],
                "frame_h": H, "frame_w": W,
                "frame_id": frame_count,
            })
        frame_count += 1

    cap.release()

    if not frame_detections:
        return

    graph = build_graph_from_detections(frame_detections, video_id, split)
    if graph is not None:
        torch.save(graph, out_path)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")  # nano = fast
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    video_paths = list(Path(args.video_dir).glob("**/*.mp4")) + \
                  list(Path(args.video_dir).glob("**/*.avi")) + \
                  list(Path(args.video_dir).glob("**/*.webm"))

    print(f"Found {len(video_paths)} videos in {args.video_dir}")
    print(f"Output → {args.out_dir}")

    from tqdm import tqdm
    for vp in tqdm(video_paths):
        try:
            process_video(str(vp), model, args.out_dir, args.split, args.fps_sample)
        except Exception as e:
            print(f"  SKIP {vp.name}: {e}")

    # Count outputs
    n_graphs = len(list(Path(args.out_dir).glob("*.pt")))
    print(f"\nBuilt {n_graphs} pseudo scene graphs.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_dir",  required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--split",      default="train")
    p.add_argument("--fps_sample", type=float, default=1.0, help="frames per second to sample")
    p.add_argument("--n_workers",  type=int,   default=1)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
