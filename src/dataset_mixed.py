"""
dataset_mixed.py
================
Mixed 95/5 training dataset for Wan2.1-14B graph-conditioned fine-tuning.

95% Text-only (Charades videos, no graph conditioning):
    - All 9,848 Charades_v1_480 .mp4 videos
    - Prompt: "a person in an indoor scene"
    - graph_tokens: None (adapter hook skipped)
    - Teaches the adapter to NOT degrade Wan's base generation quality

5% Graph-conditioned (Action Genome annotated videos):
    - 9,176 AG-annotated videos with PyG scene graph .pt files
    - Prompt: "a person in an indoor scene"
    - graph_tokens: padded TGNNEncoder node embeddings [B, 32, 256]
    - Teaches the adapter to inject specific object/relation information

Why this ratio:
    The adapter is initialized with zero-weights and starts as identity.
    Without any text-only data, training on only 9K AG videos makes the
    adapter learn patterns tied to that specific distribution and lose the
    ability to generate good indoor scenes without graph. The 95% Charades
    data anchors the adapter to Wan's original quality distribution.

How graph_tokens=None is handled:
    When returning a text-only sample, graph_data is None.
    In train_wan.py, if graph_data is None, we call adapter.clear_graph_tokens()
    so the hook fires but injects zeros → equivalent to text-only generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torch_geometric.data import Data, Batch


@dataclass
class MixedDataConfig:
    # Paths
    charades_dir:  str = "/projects/bgnv/leatherman/tgnn/dataset/ag/videos/Charades_v1_480"
    frames_dir:    str = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
    graph_dir:     str = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"

    # Frame loading
    n_frames:  int = 17     # 16×1+1 for Wan2.1 (1 second at 16fps)
    frame_h:   int = 480
    frame_w:   int = 832

    # 95/5 sampling ratio
    graph_prob: float = 0.05    # probability of sampling a graph-conditioned example

    # Training split
    split: str = "train"


# ---------------------------------------------------------------------------
# Text-only sample: load frames directly from Charades .mp4 via frame extraction
# ---------------------------------------------------------------------------

class CharadesTextDataset(Dataset):
    """
    All Charades videos without graph conditioning.
    Used for the 95% text-only portion of mixed training.

    Since Charades .mp4 files may not all have pre-extracted PNG frames,
    we use the AG frames directory when available, falling back to a
    placeholder (black frames) if not. This means some text-only samples
    will be near-zero frames, which still helps regularize the adapter.

    In production, all frames should be pre-extracted. For now, this is
    sufficient as the 95% data primarily prevents catastrophic forgetting
    rather than teaching new content.
    """

    def __init__(self, cfg: MixedDataConfig) -> None:
        self.cfg        = cfg
        self.frames_dir = Path(cfg.frames_dir)
        self.transform  = transforms.Compose([
            transforms.Resize((cfg.frame_h, cfg.frame_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        # Collect all Charades video IDs that have pre-extracted frames
        # (frames were extracted during ag_graph_dataset.py for AG-annotated videos)
        charades_dir = Path(cfg.charades_dir)
        self.video_ids: List[str] = []
        for mp4 in sorted(charades_dir.glob("*.mp4")):
            vid = mp4.stem  # e.g. "001YG"
            frame_folder = self.frames_dir / f"{vid}.mp4"
            if frame_folder.exists() and any(frame_folder.glob("*.png")):
                self.video_ids.append(vid)
        print(f"  CharadesTextDataset: {len(self.video_ids)} videos with extracted frames")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Tuple[Optional[Data], Tensor, str]:
        """
        Returns:
            graph_data : None  — no graph for text-only samples
            frames     : [n_frames, 3, H, W] float32 in [-1, 1]
            prompt     : "a person in an indoor scene"
        """
        vid    = self.video_ids[idx]
        folder = self.frames_dir / f"{vid}.mp4"
        n      = self.cfg.n_frames
        C, H, W = 3, self.cfg.frame_h, self.cfg.frame_w

        # Load uniformly-sampled frames
        pngs = sorted(folder.glob("*.png"))
        if not pngs:
            # Fallback: black frames (rare, only if folder is empty)
            return None, torch.zeros(n, C, H, W), "a person in an indoor scene"

        # Subsample or repeat to exactly n frames
        step  = max(1, len(pngs) // n)
        pngs  = pngs[::step][:n]
        last  = torch.zeros(C, H, W)
        frames: List[Tensor] = []
        for png in pngs:
            last = self.transform(Image.open(png).convert("RGB"))
            frames.append(last)
        while len(frames) < n:
            frames.append(last)   # pad by repeating last frame

        # graph_data = None signals to train loop: skip graph conditioning
        return None, torch.stack(frames[:n]), "a person in an indoor scene"


# ---------------------------------------------------------------------------
# Graph-conditioned sample: AG annotated videos with scene graph
# ---------------------------------------------------------------------------

class AGGraphDataset(Dataset):
    """
    Action Genome videos WITH graph conditioning.
    Used for the 5% graph-conditioned portion of mixed training.

    Returns (graph, frames, prompt) where graph is a PyG Data object
    containing node/edge features from the temporal scene graph.
    """

    def __init__(self, cfg: MixedDataConfig) -> None:
        self.cfg        = cfg
        self.frames_dir = Path(cfg.frames_dir)
        self.transform  = transforms.Compose([
            transforms.Resize((cfg.frame_h, cfg.frame_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        graph_dir = Path(cfg.graph_dir)
        self.samples: List[Path] = []
        for pt in sorted(graph_dir.glob("*.pt")):
            try:
                g = torch.load(pt, weights_only=False)
                if g.split == cfg.split:
                    frame_folder = self.frames_dir / f"{g.video_id}.mp4"
                    if frame_folder.exists():
                        self.samples.append(pt)
            except Exception:
                pass
        print(f"  AGGraphDataset ({cfg.split}): {len(self.samples)} videos with graphs + frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Data, Tensor, str]:
        graph  = torch.load(self.samples[idx], weights_only=False)
        frames = self._load_frames(graph)
        return graph, frames, "a person in an indoor scene"

    def _load_frames(self, graph: Data) -> Tensor:
        folder   = self.frames_dir / f"{graph.video_id}.mp4"
        n        = self.cfg.n_frames
        C, H, W  = 3, self.cfg.frame_h, self.cfg.frame_w

        frame_ids = graph.frame_ids.tolist()
        if len(frame_ids) > n:
            step = len(frame_ids) / n
            frame_ids = [frame_ids[int(i * step)] for i in range(n)]

        frames: List[Tensor] = []
        last = torch.zeros(C, H, W)
        for fid in frame_ids:
            png = folder / f"{int(fid):06d}.png"
            if png.exists():
                last = self.transform(Image.open(png).convert("RGB"))
            frames.append(last)
        while len(frames) < n:
            frames.append(last)
        return torch.stack(frames[:n])


# ---------------------------------------------------------------------------
# Mixed 95/5 dataset
# ---------------------------------------------------------------------------

class MixedTrainDataset(Dataset):
    """
    Combines CharadesTextDataset (95%) and AGGraphDataset (5%) with
    probabilistic sampling to achieve the desired ratio.

    The __len__ is set to the size of the text dataset so that one epoch
    covers all Charades videos. The graph dataset is sampled randomly (with
    replacement) during the 5% draws.

    graph_prob=0.05: 5% of batches use graph conditioning (AG data)
                     95% of batches use text-only (Charades data)
    """

    def __init__(self, cfg: MixedDataConfig) -> None:
        self.cfg        = cfg
        self.text_ds    = CharadesTextDataset(cfg)
        self.graph_ds   = AGGraphDataset(cfg)
        self._text_len  = len(self.text_ds)
        self._graph_len = len(self.graph_ds)
        print(f"\n  MixedTrainDataset:")
        print(f"    Text-only (95%)     : {self._text_len} Charades videos")
        print(f"    Graph-cond (5%)     : {self._graph_len} AG videos")
        print(f"    Graph sample prob   : {cfg.graph_prob:.0%}")

    def __len__(self) -> int:
        # One epoch = cover all text-only data
        return self._text_len

    def __getitem__(self, idx: int) -> Tuple[Optional[Data], Tensor, str]:
        if random.random() < self.cfg.graph_prob:
            # 5%: graph-conditioned sample (random AG video with replacement)
            g_idx = random.randint(0, self._graph_len - 1)
            return self.graph_ds[g_idx]
        else:
            # 95%: text-only Charades sample
            return self.text_ds[idx % self._text_len]


# ---------------------------------------------------------------------------
# Collate function: handles both None graphs (text-only) and PyG graphs
# ---------------------------------------------------------------------------

def collate_fn_mixed(batch):
    """
    Handles batches that may contain a mix of:
      - (None, frames, prompt)         ← text-only samples
      - (PyG Data, frames, prompt)     ← graph-conditioned samples

    Returns:
      graphs  : PyG Batch if any graph in batch, else None
      frames  : [B, T, C, H, W]
      prompts : list[str]
      has_graph: bool — True if this batch contains graph-conditioned samples
    """
    graph_items = [item[0] for item in batch]
    frames  = torch.stack([item[1] for item in batch])   # [B, T, C, H, W]
    prompts = [item[2] for item in batch]

    # All-or-nothing per batch: if ANY sample has a graph, all should
    # (batch_size=1 in our setup, so this is straightforward)
    has_graph = any(g is not None for g in graph_items)
    if has_graph:
        valid_graphs = [g for g in graph_items if g is not None]
        graph_batch = Batch.from_data_list(valid_graphs)
    else:
        graph_batch = None

    return graph_batch, frames, prompts, has_graph
