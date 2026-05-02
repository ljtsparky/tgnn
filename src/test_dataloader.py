"""
test_dataloader.py
==================
Smoke-test for AGGraphDataset / build_dataloader.

Checks:
  1. Dataset length matches expected train/test split counts
  2. A single __getitem__ returns correct types and shapes
  3. A DataLoader batch collates without errors
  4. Frame pixel range is [-1, 1]
  5. Graph batch vector is sane
"""

import sys
import torch
from ag_dataloader import AGGraphDataset, build_dataloader

GRAPH_DIR  = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
FRAMES_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
FRAME_SIZE = (256, 256)

def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)

# ---------------------------------------------------------------------------
print("\n=== 1. Dataset lengths ===")
train_ds = AGGraphDataset(GRAPH_DIR, FRAMES_DIR, split="train", frame_size=FRAME_SIZE)
test_ds  = AGGraphDataset(GRAPH_DIR, FRAMES_DIR, split="test",  frame_size=FRAME_SIZE)
print(f"  Train samples : {len(train_ds)}")
print(f"  Test  samples : {len(test_ds)}")
check(len(train_ds) > 5000, "train split has >5000 videos")
check(len(test_ds)  > 500,  "test  split has >500  videos")

# ---------------------------------------------------------------------------
print("\n=== 2. Single __getitem__ ===")
graph, frames = train_ds[0]
print(f"  video_id     : {graph.video_id}")
print(f"  num_frames   : {graph.num_frames}")
print(f"  frames shape : {frames.shape}")          # [T, C, H, W]
print(f"  node_class   : {graph.node_class.shape}")
print(f"  node_bbox    : {graph.node_bbox.shape}")
print(f"  edge_index   : {graph.edge_index.shape}")
check(frames.ndim == 4,                               "frames is 4-D [T,C,H,W]")
check(frames.shape[0] == graph.num_frames,            "T matches graph.num_frames")
check(frames.shape[1:] == (3, FRAME_SIZE[0], FRAME_SIZE[1]), "C,H,W correct")
check(frames.dtype == torch.float32,                  "frames dtype float32")

# ---------------------------------------------------------------------------
print("\n=== 3. Pixel range ===")
fmin, fmax = frames.min().item(), frames.max().item()
print(f"  pixel range  : [{fmin:.3f}, {fmax:.3f}]")
check(fmin >= -1.01 and fmax <= 1.01, "pixels in [-1, 1]")

# ---------------------------------------------------------------------------
print("\n=== 4. DataLoader batch (batch_size=2) ===")
loader = build_dataloader(
    GRAPH_DIR, FRAMES_DIR,
    split="train", batch_size=2, frame_size=FRAME_SIZE,
    num_workers=0, shuffle=False,
)
batch_graphs, batch_frames = next(iter(loader))
print(f"  batch graphs type   : {type(batch_graphs).__name__}")
print(f"  batch.num_graphs    : {batch_graphs.num_graphs}")
print(f"  batch vector shape  : {batch_graphs.batch.shape}")
print(f"  num frames in batch : {[f.shape[0] for f in batch_frames]}")
check(batch_graphs.num_graphs == 2,         "batch contains 2 graphs")
check(len(batch_frames) == 2,               "batch_frames list has 2 entries")
check(batch_frames[0].ndim == 4,            "each frames entry is 4-D")

# ---------------------------------------------------------------------------
print("\n=== All checks passed ===\n")
