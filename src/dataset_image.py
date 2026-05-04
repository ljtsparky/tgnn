"""
dataset_image.py
================
Image-level dataset for SDXL graph-conditioned fine-tuning.

Each training sample is:
    (target_frame_image, full_temporal_graph, target_frame_idx, prompt)

where target_frame_idx is sampled from the graph's annotated frames so the
TGNN can use temporal context to enrich the per-frame representation.

70/30 mix
---------
70% LAION-style text-only "anchor" examples (Charades random frames):
    keeps SDXL's natural distribution; prevents catastrophic forgetting
    of base image quality. graph_data = None.

30% Action Genome graph-conditioned examples:
    real temporal graph + a randomly chosen target frame from frame_ids.
    This is where the adapter actually learns.

Train/val/test split
--------------------
AG train (7427 graphs)  →  90% train + 10% val      (carved by video_id hash)
AG test  (1749 graphs)  →  held out for final eval (untouched in training)

The split is deterministic (seeded hash of video_id) so re-running gives
the same split across machines.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Batch, Data


# ---------------------------------------------------------------------------
# AG class & relation mappings (must match ag_graph_dataset.py)
# ---------------------------------------------------------------------------

_AG_CLASSES_TXT = (
    "/projects/bgnv/leatherman/tgnn/dataset/ag/annotations/"
    "action_genome_v1.0/object_classes.txt"
)
try:
    _CLASS_NAMES = [l.strip() for l in open(_AG_CLASSES_TXT) if l.strip()]
except Exception:
    _CLASS_NAMES = []

# Compound class names → human-readable singular for prompt
_NAME_TO_PROMPT = {
    "person":         "person",
    "bag":            "bag",         "bed":            "bed",
    "blanket":        "blanket",     "book":           "book",
    "box":            "box",         "broom":          "broom",
    "chair":          "chair",       "closetcabinet":  "cabinet",
    "clothes":        "clothes",     "cupglassbottle": "cup",
    "dish":           "dish",        "door":           "door",
    "doorknob":       "doorknob",    "doorway":        "doorway",
    "floor":          "floor",       "food":           "food",
    "groceries":      "groceries",   "laptop":         "laptop",
    "light":          "light",       "medicine":       "medicine bottle",
    "mirror":         "mirror",      "papernotebook":  "notebook",
    "phonecamera":    "phone",       "picture":        "picture",
    "pillow":         "pillow",      "refrigerator":   "refrigerator",
    "sandwich":       "sandwich",    "shelf":          "shelf",
    "shoe":           "shoe",        "sofacouch":      "sofa",
    "table":          "table",       "television":     "television",
    "towel":          "towel",       "vacuum":         "vacuum",
    "window":         "window",
}

# AG contacting relation names (must match ag_graph_dataset.py)
_CONTACTING = ["holding","not_contacting","touching","sitting_on","leaning_on",
               "other_relationship","standing_on","wearing","lying_on",
               "covered_by","carrying","drinking_from","eating","writing_on",
               "wiping","have_it_on_the_back","twisting"]

# Relations that read naturally as English verbs/preps
_REL_TO_VERB = {
    "holding":      "holding",
    "touching":     "touching",
    "sitting_on":   "sitting on",
    "leaning_on":   "leaning on",
    "standing_on":  "standing on",
    "wearing":      "wearing",
    "lying_on":     "lying on",
    "carrying":     "carrying",
    "drinking_from":"drinking from",
    "eating":       "eating",
    "writing_on":   "writing on",
    "wiping":       "wiping",
    # skip: not_contacting, other_relationship, covered_by, have_it_on_the_back, twisting
}

GENERIC_PROMPT = "a person in an indoor scene"


def build_prompt_from_graph(graph: Data, target_idx: int) -> str:
    """
    Build an object-aware prompt from the target frame's sub-graph.

    Format:
      "a person {verb obj}, {verb obj}, ..., with {obj}, {obj}, ..., in an indoor scene"

    If target frame has no objects/relations, falls back to GENERIC_PROMPT.
    """
    if not _CLASS_NAMES:
        return GENERIC_PROMPT

    target_mask = (graph.node_frame == target_idx)
    if int(target_mask.sum().item()) == 0:
        return GENERIC_PROMPT

    # Objects in target frame (excluding person)
    tgt_classes = sorted(set(int(c) for c in graph.node_class[target_mask].tolist()) - {0})
    if not tgt_classes:
        return GENERIC_PROMPT
    obj_words = [_NAME_TO_PROMPT[_CLASS_NAMES[c]] for c in tgt_classes]

    # Person→object relations IN target frame (only spatial edges)
    src = graph.edge_index[0]
    dst = graph.edge_index[1]
    et  = graph.edge_type
    ea  = graph.edge_attr

    rel_phrases = []
    seen = set()
    for i in range(src.size(0)):
        if int(et[i]) != 0: continue
        s, d = int(src[i]), int(dst[i])
        if not (target_mask[s] and target_mask[d]): continue
        if int(graph.node_class[s]) != 0:           continue   # source must be person
        d_cls = int(graph.node_class[d])
        if d_cls == 0:                              continue   # skip self
        # Decode contacting (cols 9..26)
        contact = ea[i, 9:26]
        for k in range(17):
            if contact[k].item() > 0.5:
                rname = _CONTACTING[k]
                verb = _REL_TO_VERB.get(rname)
                if verb is None: continue
                obj_word = _NAME_TO_PROMPT[_CLASS_NAMES[d_cls]]
                key = (verb, obj_word)
                if key in seen:                     continue
                seen.add(key)
                rel_phrases.append(f"{verb} {obj_word}")

    parts = ["a person"]
    if rel_phrases:
        parts.append(", ".join(rel_phrases[:4]))   # cap at 4 to avoid prompt overflow
    parts.append("with " + ", ".join(obj_words[:6]))
    parts.append("in an indoor scene")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ImageDataConfig:
    charades_dir:  str = "/projects/bgnv/leatherman/tgnn/dataset/ag/videos/Charades_v1_480"
    frames_dir:    str = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
    # Optional dense extraction (every-Nth raw frame from Charades). When
    # this directory contains a video's frames we use it for AG samples,
    # giving ~50x more (frame, graph) pairs vs the AG-annotated frames.
    frames_full_dir: str = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames_full"

    graph_dir:     str = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"

    # SDXL native resolution
    image_h:   int = 1024
    image_w:   int = 1024

    # Mix ratio
    graph_prob: float = 0.30

    # Split: which split to use ('train' | 'val' | 'test')
    split: str = "train"

    # Hash-based val carving from AG train graphs (deterministic)
    val_fraction: float = 0.10
    seed:         int   = 42


# ---------------------------------------------------------------------------
# Deterministic split helper
# ---------------------------------------------------------------------------

def _video_in_val(video_id: str, val_fraction: float, seed: int) -> bool:
    """Hash-based carve: same video_id always lands in the same bucket."""
    h = hashlib.md5(f"{seed}:{video_id}".encode()).hexdigest()
    bucket = int(h[:8], 16) / (1 << 32)
    return bucket < val_fraction


def _square_crop(img: Image.Image, size: int) -> Image.Image:
    """Center-crop to square then resize to (size, size). Preserves aspect."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Charades text-only dataset (random frame from each video)
# ---------------------------------------------------------------------------

class CharadesImageDataset(Dataset):
    """
    Text-only anchor: random frame from each Charades video.
    Used for 70% of training samples to keep SDXL's distribution.
    """

    def __init__(self, cfg: ImageDataConfig) -> None:
        self.cfg = cfg
        self.frames_dir = Path(cfg.frames_dir)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda im: _square_crop(im, cfg.image_h)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        graph_dir = Path(cfg.graph_dir)
        ag_video_ids = set()
        for pt in sorted(graph_dir.glob("*.pt")):
            try:
                g = torch.load(pt, weights_only=False)
                if g.split == "train":
                    ag_video_ids.add(g.video_id)
            except Exception:
                pass

        # Charades videos with extracted frames; for text-only anchor we use
        # only those NOT in AG val carve, to avoid leakage.
        charades_dir = Path(cfg.charades_dir)
        self.video_ids: List[str] = []
        for mp4 in sorted(charades_dir.glob("*.mp4")):
            vid = mp4.stem
            folder = self.frames_dir / f"{vid}.mp4"
            if not (folder.exists() and any(folder.glob("*.png"))):
                continue
            # If this video is also in AG and lands in val/test, skip
            if vid in ag_video_ids and _video_in_val(vid, cfg.val_fraction, cfg.seed):
                continue
            self.video_ids.append(vid)
        print(f"  CharadesImageDataset: {len(self.video_ids)} videos (text-only anchor)")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Tuple[Optional[Data], Tensor, int, str]:
        vid    = self.video_ids[idx]
        folder = self.frames_dir / f"{vid}.mp4"
        pngs   = sorted(folder.glob("*.png"))
        png    = random.choice(pngs)
        img    = self.transform(Image.open(png).convert("RGB"))
        # graph_data=None, target_frame_idx=-1 (unused)
        return None, img, -1, GENERIC_PROMPT


# ---------------------------------------------------------------------------
# AG graph-conditioned image dataset
# ---------------------------------------------------------------------------

class AGImageDataset(Dataset):
    """
    Each sample = (graph, target_frame_image, target_frame_idx, prompt).

    target_frame_idx: position in graph.frame_ids (0..T-1), randomly chosen
    each __getitem__ call so the same graph yields many distinct samples
    across epochs (effective augmentation).
    """

    def __init__(self, cfg: ImageDataConfig) -> None:
        self.cfg = cfg
        self.frames_dir      = Path(cfg.frames_dir)
        self.frames_full_dir = Path(cfg.frames_full_dir)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda im: _square_crop(im, cfg.image_h)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])

        graph_dir = Path(cfg.graph_dir)
        self.samples: List[Path] = []
        n_dense = 0
        for pt in sorted(graph_dir.glob("*.pt")):
            try:
                g = torch.load(pt, weights_only=False)
            except Exception:
                continue

            # Filter by split
            if cfg.split == "train":
                if g.split != "train":           continue
                if _video_in_val(g.video_id, cfg.val_fraction, cfg.seed): continue
            elif cfg.split == "val":
                if g.split != "train":           continue
                if not _video_in_val(g.video_id, cfg.val_fraction, cfg.seed): continue
            elif cfg.split == "test":
                if g.split != "test":            continue
            else:
                raise ValueError(f"Unknown split: {cfg.split}")

            folder = self.frames_dir / f"{g.video_id}.mp4"
            if not folder.exists():
                continue
            self.samples.append(pt)

            ff_folder = self.frames_full_dir / f"{g.video_id}.mp4"
            if ff_folder.exists() and any(ff_folder.glob("*.png")):
                n_dense += 1

        print(f"  AGImageDataset({cfg.split}): {len(self.samples)} videos "
              f"({n_dense} with dense frames_full)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Data, Tensor, int, str]:
        graph: Data = torch.load(self.samples[idx], weights_only=False)
        T = int(graph.num_frames)

        # Stage-B: prefer dense frames_full when available (~50x more
        # (frame, graph) pairs per video). Fall back to annotated frames.
        ff_folder = self.frames_full_dir / f"{graph.video_id}.mp4"
        ff_pngs = sorted(ff_folder.glob("*.png")) if ff_folder.exists() else []

        png = None
        if ff_pngs:
            # Random dense frame; map its frame_id to the closest annotated
            # frame so target_idx_in_list points at a real graph node bucket.
            png = random.choice(ff_pngs)
            try:
                # filenames look like 'frame_000123.png' or '000123.png'
                stem_digits = "".join(c for c in png.stem if c.isdigit())
                dense_fid = int(stem_digits) if stem_digits else 0
            except Exception:
                dense_fid = 0
            ann_ids = graph.frame_ids.tolist()
            target_idx_in_list = int(min(range(T), key=lambda i: abs(int(ann_ids[i]) - dense_fid)))
        else:
            # Pick a random target frame from the annotated frames
            target_idx_in_list = random.randrange(T)
            target_frame_id    = int(graph.frame_ids[target_idx_in_list].item())
            png = self.frames_dir / f"{graph.video_id}.mp4" / f"{target_frame_id:06d}.png"
            if not png.exists():
                cand = sorted((self.frames_dir / f"{graph.video_id}.mp4").glob("*.png"))
                if not cand:
                    img = torch.zeros(3, self.cfg.image_h, self.cfg.image_w)
                    prompt = build_prompt_from_graph(graph, target_idx_in_list)
                    return graph, img, target_idx_in_list, prompt
                png = cand[0]

        # Build object-aware prompt from the (closest) annotated target frame
        prompt = build_prompt_from_graph(graph, target_idx_in_list)
        img = self.transform(Image.open(png).convert("RGB"))
        return graph, img, target_idx_in_list, prompt


# ---------------------------------------------------------------------------
# Mixed 70/30 dataset
# ---------------------------------------------------------------------------

class MixedImageDataset(Dataset):
    """
    Train: 70% Charades (text-only) + 30% AG (graph-conditioned).
    Val/test: 100% AG graph samples (no Charades, no mix).
    """

    def __init__(self, cfg: ImageDataConfig) -> None:
        self.cfg = cfg
        self.ag = AGImageDataset(cfg)
        if cfg.split == "train":
            self.text = CharadesImageDataset(cfg)
            self._text_len = len(self.text)
        else:
            self.text = None
            self._text_len = 0
        self._ag_len = len(self.ag)

        print(f"\n  MixedImageDataset({cfg.split}):")
        if self.text is not None:
            print(f"    Text-only (70%) : {self._text_len} Charades frames")
            print(f"    Graph-cond (30%): {self._ag_len} AG videos")
        else:
            print(f"    Graph-cond only : {self._ag_len} AG videos")

    def __len__(self) -> int:
        if self.cfg.split == "train":
            # One epoch = cover all Charades anchors
            return self._text_len
        return self._ag_len

    def __getitem__(self, idx: int):
        if self.cfg.split != "train":
            return self.ag[idx]
        if random.random() < self.cfg.graph_prob:
            return self.ag[random.randrange(self._ag_len)]
        return self.text[idx % self._text_len]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn_image(batch):
    """
    Returns:
      graph_batch  : PyG Batch (only graph items) or None
      images       : [B, 3, H, W]
      target_idxs  : LongTensor [B] (-1 for text-only items)
      prompts      : list[str]
      has_graph_mask : BoolTensor [B] — True where item has a graph
    """
    graphs   = [b[0] for b in batch]
    images   = torch.stack([b[1] for b in batch])
    targets  = torch.tensor([b[2] for b in batch], dtype=torch.long)
    prompts  = [b[3] for b in batch]

    has_graph_mask = torch.tensor([g is not None for g in graphs], dtype=torch.bool)
    valid = [g for g in graphs if g is not None]
    graph_batch = Batch.from_data_list(valid) if valid else None

    return graph_batch, images, targets, prompts, has_graph_mask
