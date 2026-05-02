"""
ag_dataloader.py
=================
PyTorch Dataset + DataLoader for Action Genome temporal scene graphs.

Each sample is a (graph, frames_tensor) pair:
  graph          PyG Data object  — as built by ag_graph_dataset.py
  frames_tensor  [T, C, H, W]     float32, values in [-1, 1]
                                   T = graph.num_frames valid annotated frames

Usage:
    from ag_dataloader import AGGraphDataset, build_dataloader

    train_loader = build_dataloader(
        graph_dir  = '/projects/bgnv/leatherman/tgnn/dataset/ag/graphs',
        frames_dir = '/projects/bgnv/leatherman/tgnn/dataset/ag/frames',
        split      = 'train',
        batch_size = 4,
        frame_size = (256, 256),
        num_workers= 4,
    )

    for graphs, frames in train_loader:
        # graphs : PyG Batch object  (all nodes/edges concatenated)
        # frames : list of [T_i, C, H, W] tensors  (variable T per video)
        ...

Notes on collation
------------------
Graphs are batched with torch_geometric.data.Batch.from_data_list — standard
PyG batching that adds a `batch` vector for scatter operations.

Frames are NOT padded into a single tensor because T varies per video (2-126).
The collate_fn returns them as a plain list of [T, C, H, W] tensors.
If your training loop needs a fixed-T batch, subsample T frames inside the loop
(e.g. uniform T=16 sampling) rather than here — keeps the Dataset generic.

Missing frames
--------------
If a frame PNG is absent on disk (should not happen after the full ffmpeg dump,
but defensive), it is replaced with a zero tensor and a warning is printed once
per video. The graph is still returned intact.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch_geometric.data import Batch, Data


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AGGraphDataset(Dataset):
    """
    Parameters
    ----------
    graph_dir   : directory with VIDEO_ID.pt files
    frames_dir  : directory with VIDEO_ID.mp4/XXXXXX.png files
    split       : 'train' | 'test' | None (keep all)
    frame_size  : (H, W) to resize every frame
    """

    def __init__(
        self,
        graph_dir:  str,
        frames_dir: str,
        split:      Optional[str] = "train",
        frame_size: Tuple[int, int] = (256, 256),
    ) -> None:
        self.graph_dir  = Path(graph_dir)
        self.frames_dir = Path(frames_dir)
        self.frame_size = frame_size

        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.ToTensor(),                      # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5],       # → [-1, 1]
                                  [0.5, 0.5, 0.5]),
        ])

        # Collect all .pt paths, optionally filter by split
        all_pts = sorted(self.graph_dir.glob("*.pt"))
        if split is not None:
            filtered = []
            for pt in all_pts:
                try:
                    data = torch.load(pt, weights_only=False)
                    if data.split == split:
                        filtered.append(pt)
                except Exception:
                    pass
            self.pt_files = filtered
        else:
            self.pt_files = all_pts

        self._missing_warned: set = set()   # suppress repeated warnings per video

    def __len__(self) -> int:
        return len(self.pt_files)

    def __getitem__(self, idx: int) -> Tuple[Data, torch.Tensor]:
        pt_path = self.pt_files[idx]
        graph   = torch.load(pt_path, weights_only=False)

        frames_tensor = self._load_frames(graph.video_id, graph.frame_ids)
        return graph, frames_tensor

    # ------------------------------------------------------------------
    def _load_frames(self, video_id: str, frame_ids: torch.Tensor) -> torch.Tensor:
        """
        Load PNG frames for this video and return [T, C, H, W] float32 [-1,1].
        """
        folder = self.frames_dir / f"{video_id}.mp4"
        C, H, W = 3, self.frame_size[0], self.frame_size[1]
        frames: List[torch.Tensor] = []

        for fid in frame_ids.tolist():
            png_path = folder / f"{int(fid):06d}.png"
            if png_path.exists():
                img = Image.open(png_path).convert("RGB")
                frames.append(self.transform(img))
            else:
                # Missing frame: fill with zeros, warn once per video
                if video_id not in self._missing_warned:
                    warnings.warn(
                        f"Frame not found: {png_path}  (filling with zeros)",
                        stacklevel=2,
                    )
                    self._missing_warned.add(video_id)
                frames.append(torch.zeros(C, H, W, dtype=torch.float32))

        return torch.stack(frames)   # [T, C, H, W]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[Data, torch.Tensor]]
) -> Tuple[Batch, List[torch.Tensor]]:
    """
    graphs : PyG Batch (node_class, node_bbox, node_frame, edge_index, etc.)
    frames : list of [T_i, C, H, W]  — one tensor per video in the batch
    """
    graphs = [item[0] for item in batch]
    frames = [item[1] for item in batch]
    return Batch.from_data_list(graphs), frames


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_dataloader(
    graph_dir:   str,
    frames_dir:  str,
    split:       Optional[str] = "train",
    batch_size:  int = 4,
    frame_size:  Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    shuffle:     bool = True,
) -> DataLoader:
    dataset = AGGraphDataset(
        graph_dir  = graph_dir,
        frames_dir = frames_dir,
        split      = split,
        frame_size = frame_size,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = True,
    )
