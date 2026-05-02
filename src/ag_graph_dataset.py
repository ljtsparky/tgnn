"""
ag_graph_dataset.py
====================
Builds per-video temporal scene graphs from Action Genome annotations
and serializes them as PyTorch Geometric Data objects (.pt files).

Usage (build all graphs):
    python ag_graph_dataset.py \
        --ann_dir /projects/bgnv/leatherman/tgnn/dataset/ag/annotations/action_genome_v1.0 \
        --out_dir /projects/bgnv/leatherman/tgnn/dataset/ag/graphs

Each output file  graphs/VIDEO_ID.pt  contains a single PyG Data object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Node class index convention
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  node_class = 0          → person  (fixed; never appears in object_classes.txt)
  node_class = 1 … 35    → object classes, corresponding to lines 1-35 of
                            object_classes.txt  (line 0 = __background__, always skipped)

  Example: object_classes.txt line 3 = "blanket"  → node_class = 3
  There is no offset arithmetic; the line number IS the node_class index.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Node features stored (raw — embedding happens inside T-GNN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  node_class  [N]     int64    class index as above
  node_bbox   [N, 4]  float32  (x, y, w, h) normalised to [0,1] per-frame resolution
                               resolution comes from person_bbox.pkl's bbox_size field
                               (NOT hardcoded — Charades has 20+ different resolutions)
  node_frame  [N]     int64    which annotated frame this node belongs to (0-indexed
                               over the valid frames in this video)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Edge layout
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  All edges live in one edge_index / edge_attr / edge_type triple.

  Spatial edges  (edge_type == 0):
      person -> object, within one frame.
      edge_attr: [attn_multihot(3) | spatial_multihot(6) | contact_multihot(17) | iou=1.0 | gap=0.0]
      Width: 3 + 6 + 17 + 1 + 1 = 28

  Temporal edges (edge_type == 1):
      matched node in frame t  ->  matched node in frame t+1.
      edge_attr: [zeros(26) | iou | frame_gap_normalised]
      Same width (28); relationship slots are zero for temporal edges.

  Matching rules:
    - Person is always connected across consecutive valid frames.
    - Objects are matched greedily by class-constrained bbox IoU.
    - A match requires IoU >= IOU_THRESHOLD (default 0.3, configurable).
    - Actual IoU is stored in edge_attr[26] so T-GNN can soft-weight matches.
    - frame_gap_normalised = raw_frame_gap / MAX_FRAME_GAP, clipped to [0, 1].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frames with no person detection (14% of frames)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  person_bbox.pkl has 40,482 frames where bbox.shape == (0, 4).
  Decision: SKIP the entire frame (do not fill with previous bbox).
  Reason: spatial edges are person->object; a fabricated person bbox would make
  node_bbox features unreliable. The video graph simply omits those frames.
  If this reduces a video to <2 valid frames, the whole video is skipped.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Graph-level metadata
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  video_id    str         e.g. '0A8CF'
  frame_ids   [T] int64   original frame numbers of the valid frames
  num_frames  int         T  (number of valid annotated frames kept)
  split       str         'train' or 'test'  (from annotation metadata)
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

IOU_THRESHOLD = 0.3    # minimum IoU to form a temporal edge between objects
                        # actual IoU is stored in edge_attr regardless
MAX_FRAME_GAP = 150.0  # normalisation denominator for frame gap in edge_attr

# edge_attr column layout  (total width = 28)
ATTN_START  = 0;   ATTN_END  = 3    # [0, 3)   attention    (3 classes)
SPAT_START  = 3;   SPAT_END  = 9    # [3, 9)   spatial      (6 classes)
CONT_START  = 9;   CONT_END  = 26   # [9, 26)  contacting  (17 classes)
IOU_IDX     = 26                    # [26]     IoU value
GAP_IDX     = 27                    # [27]     normalised frame gap
EDGE_ATTR_DIM = 28

# node_class: 0 = person (fixed), 1-35 = line number in object_classes.txt
PERSON_CLASS_IDX = 0

# Relationship string lists — order defines column position in edge_attr
ATTENTION_CLASSES = ["looking_at", "not_looking_at", "unsure"]
SPATIAL_CLASSES   = ["in_front_of", "behind", "above", "beneath", "in", "on_the_side_of"]
CONTACTING_CLASSES = [
    "holding", "not_contacting", "touching", "sitting_on", "leaning_on",
    "other_relationship", "standing_on", "wearing", "lying_on", "covered_by",
    "carrying", "drinking_from", "eating", "writing_on", "wiping",
    "have_it_on_the_back", "twisting",
]


# ---------------------------------------------------------------------------
# Load class name -> index maps
# ---------------------------------------------------------------------------

def load_class_maps(ann_dir: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    obj_name2idx  : object class name -> node_class index (= line number in txt, 1-35)
                    line 0 (__background__) is skipped and never assigned an index.
    attn_name2idx : attention rel name  -> column offset within [0, 3)
    spat_name2idx : spatial rel name    -> column offset within [0, 6)
    cont_name2idx : contacting rel name -> column offset within [0, 17)
    """
    obj_path = os.path.join(ann_dir, "object_classes.txt")
    with open(obj_path) as f:
        obj_lines = [l.strip() for l in f if l.strip()]

    # line 0 = "__background__" -> skip.
    # line k (k >= 1) -> node_class = k  (no arithmetic, line number IS the index)
    obj_name2idx: Dict[str, int] = {}
    for line_no, name in enumerate(obj_lines):
        if name == "__background__":
            continue
        obj_name2idx[name] = line_no   # 1-indexed naturally

    attn_name2idx = {n: i for i, n in enumerate(ATTENTION_CLASSES)}
    spat_name2idx = {n: i for i, n in enumerate(SPATIAL_CLASSES)}
    cont_name2idx = {n: i for i, n in enumerate(CONTACTING_CLASSES)}

    return obj_name2idx, attn_name2idx, spat_name2idx, cont_name2idx


# ---------------------------------------------------------------------------
# bbox helpers
# ---------------------------------------------------------------------------

def normalise_xywh(
    x: float, y: float, w: float, h: float,
    frame_w: int, frame_h: int,
) -> List[float]:
    """Normalise (x, y, w, h) pixel bbox to [0, 1] using per-frame resolution."""
    return [x / frame_w, y / frame_h, w / frame_w, h / frame_h]


def xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    """Convert xyxy (person_bbox.pkl format) to xywh (object pkl format)."""
    return x1, y1, x2 - x1, y2 - y1


def bbox_iou_xywh(b1: Tuple, b2: Tuple) -> float:
    """IoU between two (x, y, w, h) bboxes in the same pixel coordinate space."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# FrameSnapshot: parsed data for one annotated frame
# ---------------------------------------------------------------------------

class FrameSnapshot:
    """
    node_class[i]  int    0 = person, 1-35 = object class (line number)
    node_bbox[i]   tuple  (x, y, w, h) raw pixel coords — normalised later
    frame_wh       tuple  (width, height) from person_bbox.pkl bbox_size
    edges          list   (src_local, dst_local, edge_attr_np) — spatial only
    """
    __slots__ = ("node_class", "node_bbox", "frame_wh", "edges")

    def __init__(self) -> None:
        self.node_class: List[int]   = []
        self.node_bbox:  List[Tuple] = []
        self.frame_wh:   Tuple       = (480, 270)   # overwritten immediately
        self.edges:      List[Tuple] = []


def parse_frame(
    frame_key: str,
    obj_list: List[dict],
    person_val: dict,
    obj_name2idx: Dict[str, int],
    attn_name2idx: Dict[str, int],
    spat_name2idx: Dict[str, int],
    cont_name2idx: Dict[str, int],
) -> Optional[FrameSnapshot]:
    """
    Parse one annotated frame into a FrameSnapshot.
    Returns None if person bbox is absent (frame is skipped entirely).
    """
    # ── guard: person detection absent for this frame ────────────────────────
    # person_val["bbox"] is ndarray shape (N, 4), N == 0 for 14% of frames.
    person_bboxes = person_val["bbox"]
    if person_bboxes.shape[0] == 0:
        # No person detected. Skipping entire frame.
        # We do NOT fill with previous bbox — that would corrupt node_bbox.
        return None

    # ── per-frame resolution (NOT hardcoded — Charades has 20+ resolutions) ──
    # person_bbox.pkl stores the actual frame size used for detection.
    frame_w, frame_h = person_val["bbox_size"]   # (width, height)

    snap = FrameSnapshot()
    snap.frame_wh = (frame_w, frame_h)

    # ── person node (always index 0 in local numbering) ──────────────────────
    # person_bbox.pkl uses xyxy; convert to xywh to match object pkl format
    x1, y1, x2, y2 = (float(v) for v in person_bboxes[0])
    px, py, pw, ph  = xyxy_to_xywh(x1, y1, x2, y2)
    snap.node_class.append(PERSON_CLASS_IDX)   # 0
    snap.node_bbox.append((px, py, pw, ph))    # raw pixels; normalised in build_video_graph

    # ── object nodes + spatial edges ─────────────────────────────────────────
    for obj in obj_list:
        if not obj.get("visible", False):
            # visible=False: ALL other fields are None — must skip before any access
            continue

        cls_name = obj.get("class")
        bbox     = obj.get("bbox")   # (x, y, w, h) pixels; xywh format
        if cls_name is None or bbox is None:
            continue
        if cls_name not in obj_name2idx:
            continue

        local_obj_idx = len(snap.node_class)
        snap.node_class.append(obj_name2idx[cls_name])    # line number, 1-35
        snap.node_bbox.append(tuple(float(v) for v in bbox))

        # spatial edge: person(0) -> object(local_obj_idx)
        attr = np.zeros(EDGE_ATTR_DIM, dtype=np.float32)
        attr[IOU_IDX] = 1.0    # same frame = perfect "overlap"
        attr[GAP_IDX] = 0.0    # zero temporal gap

        for r in (obj.get("attention_relationship") or []):
            if r in attn_name2idx:
                attr[ATTN_START + attn_name2idx[r]] = 1.0

        for r in (obj.get("spatial_relationship") or []):
            if r in spat_name2idx:
                attr[SPAT_START + spat_name2idx[r]] = 1.0

        for r in (obj.get("contacting_relationship") or []):
            if r in cont_name2idx:
                attr[CONT_START + cont_name2idx[r]] = 1.0

        snap.edges.append((0, local_obj_idx, attr))

    return snap


# ---------------------------------------------------------------------------
# Per-video graph builder
# ---------------------------------------------------------------------------

def build_video_graph(
    video_id: str,
    obj_data: dict,
    person_bbox_data: dict,
    obj_name2idx: Dict[str, int],
    attn_name2idx: Dict[str, int],
    spat_name2idx: Dict[str, int],
    cont_name2idx: Dict[str, int],
) -> Optional[Data]:
    """
    Build a single PyG Data object for one video.
    Returns None if the video has fewer than 2 valid annotated frames.
    """
    prefix     = video_id + ".mp4/"
    frame_keys = sorted(
        (k for k in obj_data if k.startswith(prefix)),
        key=lambda k: int(k.split("/")[1].replace(".png", ""))
    )

    if len(frame_keys) < 2:
        return None

    # ── parse frames ──────────────────────────────────────────────────────────
    snapshots:        List[FrameSnapshot] = []
    valid_frame_keys: List[str]           = []

    for fk in frame_keys:
        if fk not in person_bbox_data:
            continue
        snap = parse_frame(
            fk, obj_data[fk], person_bbox_data[fk],
            obj_name2idx, attn_name2idx, spat_name2idx, cont_name2idx,
        )
        # require at least person + 1 object so spatial edges are non-empty
        if snap is None or len(snap.node_class) < 2:
            continue
        snapshots.append(snap)
        valid_frame_keys.append(fk)

    if len(snapshots) < 2:
        return None

    # ── accumulate global nodes ───────────────────────────────────────────────
    global_node_class: List[int]  = []
    global_node_bbox:  List[List] = []   # already normalised
    global_node_frame: List[int]  = []
    snapshot_offset:   List[int]  = []   # global idx of first node in snapshot t

    for t, snap in enumerate(snapshots):
        snapshot_offset.append(len(global_node_class))
        fw, fh = snap.frame_wh
        for cls, (x, y, w, h) in zip(snap.node_class, snap.node_bbox):
            global_node_class.append(cls)
            global_node_bbox.append(normalise_xywh(x, y, w, h, fw, fh))
            global_node_frame.append(t)

    # ── collect edges ─────────────────────────────────────────────────────────
    edge_src:  List[int]        = []
    edge_dst:  List[int]        = []
    edge_attr: List[np.ndarray] = []
    edge_type: List[int]        = []   # 0 = spatial, 1 = temporal

    # Spatial (within-frame)
    for t, snap in enumerate(snapshots):
        off = snapshot_offset[t]
        for (lsrc, ldst, attr) in snap.edges:
            edge_src.append(off + lsrc)
            edge_dst.append(off + ldst)
            edge_attr.append(attr)
            edge_type.append(0)

    # Temporal (across consecutive valid frames)
    for t in range(len(snapshots) - 1):
        snap_t  = snapshots[t]
        snap_t1 = snapshots[t + 1]
        off_t   = snapshot_offset[t]
        off_t1  = snapshot_offset[t + 1]

        fnum_t  = int(valid_frame_keys[t].split("/")[1].replace(".png", ""))
        fnum_t1 = int(valid_frame_keys[t + 1].split("/")[1].replace(".png", ""))
        gap_norm = min((fnum_t1 - fnum_t) / MAX_FRAME_GAP, 1.0)

        # person -> person (always connected across frames)
        attr_p = np.zeros(EDGE_ATTR_DIM, dtype=np.float32)
        attr_p[IOU_IDX] = bbox_iou_xywh(snap_t.node_bbox[0], snap_t1.node_bbox[0])
        attr_p[GAP_IDX] = gap_norm
        edge_src.append(off_t);   edge_dst.append(off_t1)
        edge_attr.append(attr_p); edge_type.append(1)

        # object -> object (greedy class-constrained IoU matching)
        objs_t  = [(li, snap_t.node_class[li],  snap_t.node_bbox[li])
                   for li in range(1, len(snap_t.node_class))]
        objs_t1 = [(li, snap_t1.node_class[li], snap_t1.node_bbox[li])
                   for li in range(1, len(snap_t1.node_class))]

        matched_t1: set = set()
        for (li_t, cls_t, bb_t) in objs_t:
            best_iou, best_li = IOU_THRESHOLD - 1e-9, None
            for (li_t1, cls_t1, bb_t1) in objs_t1:
                if cls_t1 != cls_t or li_t1 in matched_t1:
                    continue
                iou = bbox_iou_xywh(bb_t, bb_t1)
                if iou > best_iou:
                    best_iou, best_li = iou, li_t1

            if best_li is not None:
                matched_t1.add(best_li)
                attr_o = np.zeros(EDGE_ATTR_DIM, dtype=np.float32)
                attr_o[IOU_IDX] = best_iou
                attr_o[GAP_IDX] = gap_norm
                edge_src.append(off_t + li_t)
                edge_dst.append(off_t1 + best_li)
                edge_attr.append(attr_o)
                edge_type.append(1)

    # ── build tensors ─────────────────────────────────────────────────────────
    node_class_t = torch.tensor(global_node_class, dtype=torch.long)
    node_bbox_t  = torch.tensor(global_node_bbox,  dtype=torch.float32)
    node_frame_t = torch.tensor(global_node_frame, dtype=torch.long)

    if edge_src:
        edge_index_t = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_t  = torch.tensor(np.stack(edge_attr),  dtype=torch.float32)
        edge_type_t  = torch.tensor(edge_type,            dtype=torch.long)
    else:
        edge_index_t = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t  = torch.zeros((0, EDGE_ATTR_DIM),    dtype=torch.float32)
        edge_type_t  = torch.zeros((0,),                  dtype=torch.long)

    frame_ids_t = torch.tensor(
        [int(fk.split("/")[1].replace(".png", "")) for fk in valid_frame_keys],
        dtype=torch.long,
    )

    # Infer split from first visible object's metadata
    split = "train"
    for fk in valid_frame_keys:
        for obj in obj_data.get(fk, []):
            if obj.get("visible") and obj.get("metadata"):
                split = obj["metadata"].get("set", "train")
                break
        else:
            continue
        break

    return Data(
        node_class = node_class_t,
        node_bbox  = node_bbox_t,
        node_frame = node_frame_t,
        edge_index = edge_index_t,
        edge_attr  = edge_attr_t,
        edge_type  = edge_type_t,
        frame_ids  = frame_ids_t,
        num_frames = len(snapshots),
        video_id   = video_id,
        split      = split,
    )


# ---------------------------------------------------------------------------
# Build all graphs
# ---------------------------------------------------------------------------

def build_all_graphs(ann_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print("Loading annotation pkl files ...")
    with open(os.path.join(ann_dir, "object_bbox_and_relationship.pkl"), "rb") as f:
        obj_data = pickle.load(f)
    with open(os.path.join(ann_dir, "person_bbox.pkl"), "rb") as f:
        person_bbox_data = pickle.load(f)

    obj_name2idx, attn_name2idx, spat_name2idx, cont_name2idx = load_class_maps(ann_dir)
    print(f"  object classes  : {len(obj_name2idx)}")
    print(f"  attention rels  : {len(attn_name2idx)}")
    print(f"  spatial rels    : {len(spat_name2idx)}")
    print(f"  contacting rels : {len(cont_name2idx)}")

    video_ids = sorted({k.split("/")[0].replace(".mp4", "") for k in obj_data})
    print(f"  videos to build : {len(video_ids)}")

    n_saved = n_skipped = n_existing = 0
    for vid in tqdm(video_ids, desc="Building graphs"):
        out_path = os.path.join(out_dir, f"{vid}.pt")
        if os.path.exists(out_path):
            n_existing += 1
            continue   # resume-safe

        data = build_video_graph(
            vid, obj_data, person_bbox_data,
            obj_name2idx, attn_name2idx, spat_name2idx, cont_name2idx,
        )
        if data is None:
            n_skipped += 1
            continue

        torch.save(data, out_path)
        n_saved += 1

    print(f"\nDone.")
    print(f"  Saved    : {n_saved}")
    print(f"  Skipped  : {n_skipped}  (< 2 valid frames after person-detection filter)")
    print(f"  Existing : {n_existing}  (already built, skipped for resume)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Action Genome temporal scene graphs")
    parser.add_argument(
        "--ann_dir",
        default="/projects/bgnv/leatherman/tgnn/dataset/ag/annotations/action_genome_v1.0",
    )
    parser.add_argument(
        "--out_dir",
        default="/projects/bgnv/leatherman/tgnn/dataset/ag/graphs",
    )
    args = parser.parse_args()
    build_all_graphs(args.ann_dir, args.out_dir)