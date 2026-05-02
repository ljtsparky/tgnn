"""
extract_all_frames.py
======================
Extract every Nth frame from every Charades AG-train video into
frames_full/<video_id>.mp4/<frame_idx:06d>.png

We don't need every frame at 30fps — that's overkill. Sampling every
3rd frame (~10 fps) is plenty for image conditioning while keeping
disk + decode reasonable.

Usage:
    python extract_all_frames.py --stride 3 --workers 8
"""
import argparse, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import imageio.v3 as iio
from PIL import Image

CHARADES_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/videos/Charades_v1_480"
GRAPH_DIR    = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
OUT_DIR      = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames_full"


def get_train_video_ids():
    ids = []
    for p in sorted(Path(GRAPH_DIR).glob("*.pt")):
        try:
            g = torch.load(p, weights_only=False)
            if g.split == "train":
                ids.append(g.video_id)
        except Exception:
            pass
    return ids


def extract_one(args):
    vid, stride, max_size = args
    src = Path(CHARADES_DIR) / f"{vid}.mp4"
    out = Path(OUT_DIR) / f"{vid}.mp4"
    out.mkdir(parents=True, exist_ok=True)
    # Skip if already done (count files)
    existing = list(out.glob("*.png"))
    if len(existing) >= 50:
        return f"{vid}: skip ({len(existing)} frames already)"

    try:
        # Stream frames lazily — don't load all into memory
        n_saved = 0
        for idx, frame in enumerate(iio.imiter(str(src), plugin="pyav")):
            if idx % stride != 0:
                continue
            img = Image.fromarray(frame)
            # Resize down for storage — SDXL anyway crops to 1024 sq
            if max_size and max(img.size) > max_size:
                ratio = max_size / max(img.size)
                img = img.resize((int(img.width*ratio), int(img.height*ratio)), Image.LANCZOS)
            img.save(out / f"{idx:06d}.png", optimize=True)
            n_saved += 1
        return f"{vid}: {n_saved} frames"
    except Exception as e:
        return f"{vid}: ERROR {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stride",   type=int, default=3, help="every Nth frame")
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--max_size", type=int, default=720, help="resize to fit this; 0=keep")
    p.add_argument("--limit",    type=int, default=0, help="limit # videos (0=all)")
    args = p.parse_args()

    ids = get_train_video_ids()
    if args.limit:
        ids = ids[:args.limit]
    print(f"Extracting from {len(ids)} train videos | stride={args.stride} | "
          f"workers={args.workers} | max_size={args.max_size}")
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(extract_one, (vid, args.stride, args.max_size)) for vid in ids]
        for f in as_completed(futs):
            msg = f.result()
            n_done += 1
            if n_done % 50 == 0 or "ERROR" in msg:
                print(f"[{n_done}/{len(ids)}] {msg}", flush=True)
    print(f"\nDone: {n_done}/{len(ids)}")


if __name__ == "__main__":
    main()
