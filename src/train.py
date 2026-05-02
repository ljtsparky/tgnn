"""
train.py
========
Stage 2: frozen CogVideoX-2b backbone + trainable TGNNEncoder + GraphCondAdapter.

What is frozen  : VAE, T5 text encoder, CogVideoX transformer (all parameters)
What is trained : TGNNEncoder (796K params) + GraphCondAdapter (17M params)

Training objective
------------------
Standard ε-prediction diffusion loss:
    latents  = VAE.encode(frames) * scaling_factor
    ε        ~ N(0, I)
    t        ~ Uniform[0, T_max)
    x_t      = sqrt(ᾱ_t) * latents + sqrt(1 - ᾱ_t) * ε
    ε̂        = transformer(x_t, encoder_hidden_states=cond, t)
    loss     = MSE(ε̂, ε)
where cond = cat([graph_tokens, text_emb], dim=1).

Data pipeline
-------------
For each video, we load 49 consecutive pixel frames from the raw .mp4 file,
uniformly sampled within the annotated frame range (from graph.frame_ids).
This ensures the video clip is within the region where the scene graph applies.
The graph is used as structural conditioning regardless of which exact 49 frames
are sampled — annotation density in Action Genome is sufficient (~24 frames/video
covering the key interactions).

Memory notes
------------
At fp16, batch_size=1, 480×720, 49 frames:
  VAE encode      : ~1 GB peak (done under torch.no_grad)
  Transformer fwd : ~15 GB    (frozen, but activations needed for backward
                                through the adapter → encoder_hidden_states path)
  Adapter + TGNN  : ~0.1 GB
Total estimate    : ~18 GB → fits on A100 40GB with gradient checkpointing.

If OOM: reduce frame_h/frame_w in TrainConfig (e.g., 256×256),
or enable transformer.enable_gradient_checkpointing().

Text prompt
-----------
Constructed from the graph's object classes:
  "a person [dominant relationship] [unique objects] indoors"
Example: "a person sitting on a chair with a table nearby"
This gives the T5 encoder some scene-level context while the graph tokens
provide the precise relational structure.

Usage
-----
    # Single GPU
    python train.py --out_dir /projects/bgnv/leatherman/tgnn/checkpoints

    # Resume
    python train.py --out_dir .../checkpoints --resume
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from torch_geometric.data import Batch, Data
from diffusers import CogVideoXPipeline
from diffusers import DDPMScheduler
from transformers import T5EncoderModel, T5Tokenizer

from tgnn_model import TGNNEncoder
from adapter import GraphCrossAttnAdapter, pad_node_embeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Paths
    pretrained_model: str  = "THUDM/CogVideoX-2b"
    graph_dir:        str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
    frames_dir:       str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
    out_dir:          str  = "/projects/bgnv/leatherman/tgnn/checkpoints"
    hf_cache:         str  = "/projects/bgnv/leatherman/hf_cache"

    # Data — uses pre-extracted annotated PNGs (dataset/ag/frames/), NOT raw mp4.
    # VideoReader requires torchvision compiled with ffmpeg which is not guaranteed.
    # Annotated frames are perfectly graph-aligned and already on disk.
    split:       str = "train"
    n_frames:    int = 49         # sample up to this many frames; pad by repeating last
    frame_h:     int = 480        # resize target height (reduce to 256 if OOM)
    frame_w:     int = 720        # resize target width  (reduce to 256 if OOM)
    num_workers: int = 4

    # Training
    n_steps:        int   = 10_000
    batch_size:     int   = 1
    grad_accum:     int   = 4      # effective batch = batch_size * grad_accum
    lr:             float = 1e-4
    weight_decay:   float = 0.01
    max_grad_norm:  float = 1.0
    warmup_steps:   int   = 200
    mixed_precision: str  = "fp16"   # "fp16" | "bf16" | "no"

    # Model
    d_model:        int = 256
    n_graph_tokens: int = 32    # node tokens per graph fed to cross-attn
    d_cross:        int = 256   # cross-attention inner dim
    n_cross_heads:  int = 8     # cross-attention heads
    n_gnn_layers:   int = 3
    n_gnn_heads:    int = 8
    dropout:        float = 0.1

    # Noise schedule
    n_timesteps:   int = 1000

    # Logging / saving
    log_every:     int = 20
    save_every:    int = 500
    resume:        bool = False


# ---------------------------------------------------------------------------
# Object class names (must match ag_graph_dataset.py line ordering)
# ---------------------------------------------------------------------------

_OBJECT_NAMES = [
    "__background__",
    "bag", "bed", "blanket", "book", "box", "broom", "chair", "closet/cabinet",
    "clothes", "cup/glass/bottle", "dish", "door", "doorknob", "doorway",
    "floor", "food", "groceries", "laptop", "light", "medicine",
    "mirror", "paper/notebook", "phone/camera", "picture", "pillow",
    "refrigerator", "sandwich", "shelf", "shoe", "sofa/couch",
    "table", "television", "towel", "vacuum", "window",
]

_DOMINANT_CONTACTING = [
    "holding", "not_contacting", "touching", "sitting_on", "leaning_on",
    "other_relationship", "standing_on", "wearing", "lying_on", "covered_by",
    "carrying", "drinking_from", "eating", "writing_on", "wiping",
    "have_it_on_the_back", "twisting",
]
# edge_attr contacting slots: indices 9-25
_CONT_SLICE = slice(9, 26)


def graph_to_prompt(graph: Data) -> str:
    # Generic prompt: object names intentionally omitted so that object
    # recall in the generated video can only come from graph conditioning,
    # not from T5 encoding of object class names.
    return "a person in an indoor scene"


# ---------------------------------------------------------------------------
# Training Dataset
# ---------------------------------------------------------------------------

class AGTrainDataset(Dataset):
    """
    Returns (graph, frames_tensor, prompt) for each video.

    frames_tensor: [n_frames, 3, H, W] float32 in [-1, 1]
    Frames are 49 consecutive pixel frames loaded from the .mp4, sampled
    within the annotated frame range stored in graph.frame_ids.
    """

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg        = cfg
        self.frames_dir = Path(cfg.frames_dir)
        self.transform  = transforms.Compose([
            transforms.Resize((cfg.frame_h, cfg.frame_w)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Load graph paths for the requested split; require the frames folder to exist
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

        print(f"  AGTrainDataset ({cfg.split}): {len(self.samples)} videos with frames")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Data, Tensor, str]:
        graph = torch.load(self.samples[idx], weights_only=False)
        frames = self._load_frames(graph)
        prompt = graph_to_prompt(graph)
        return graph, frames, prompt

    def _load_frames(self, graph: Data) -> Tensor:
        """
        Load frames from pre-extracted annotated PNGs in dataset/ag/frames/.
        These are the exact frames that have scene-graph annotations, so
        graph–video alignment is perfect by construction.

        If the video has fewer than n_frames annotated frames, the last
        frame is repeated to reach exactly n_frames. If it has more,
        n_frames are sampled uniformly.
        """
        from PIL import Image

        folder = self.frames_dir / f"{graph.video_id}.mp4"
        n = self.cfg.n_frames
        C, H, W = 3, self.cfg.frame_h, self.cfg.frame_w

        # frame_ids is sorted (ag_graph_dataset guarantees sorted order)
        frame_ids = graph.frame_ids.tolist()

        # Uniform subsample to at most n frames
        if len(frame_ids) > n:
            step = len(frame_ids) / n
            frame_ids = [frame_ids[int(i * step)] for i in range(n)]

        frames_out: List[Tensor] = []
        last_tensor = torch.zeros(C, H, W, dtype=torch.float32)

        for fid in frame_ids:
            png = folder / f"{int(fid):06d}.png"
            if png.exists():
                img = Image.open(png).convert("RGB")
                last_tensor = self.transform(img)
            # missing frame: reuse previous (rare, warn already done in graph builder)
            frames_out.append(last_tensor)

        # Pad by repeating last frame to reach exactly n_frames
        while len(frames_out) < n:
            frames_out.append(last_tensor)

        return torch.stack(frames_out[:n])   # [n_frames, C, H, W]


# ---------------------------------------------------------------------------
# Collate: graph batch + frame stack + prompts list
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[Data, Tensor, str]]
) -> Tuple[Batch, Tensor, List[str]]:
    graphs  = [item[0] for item in batch]
    frames  = torch.stack([item[1] for item in batch])   # [B, T, C, H, W]
    prompts = [item[2] for item in batch]
    return Batch.from_data_list(graphs), frames, prompts


# ---------------------------------------------------------------------------
# Encode text with T5 (frozen)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt(
    prompts: List[str],
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    max_length: int = 226,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
) -> Tensor:
    tokens = tokenizer(
        prompts,
        padding     = "max_length",
        max_length  = max_length,
        truncation  = True,
        return_tensors = "pt",
    )
    ids      = tokens.input_ids.to(device)
    mask     = tokens.attention_mask.to(device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        emb = text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    return emb.to(dtype)   # [B, max_length, 4096]


# ---------------------------------------------------------------------------
# Encode video frames with VAE (frozen)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_frames(
    frames: Tensor,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    frames : [B, T, C, H, W] float32 in [-1, 1]
    returns: [B, T_lat, C_lat, H_lat, W_lat] latents
    """
    # VAE expects [B, C, T, H, W]
    x = frames.to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)
    with torch.autocast(device_type=device.type, dtype=dtype):
        posterior = vae.encode(x).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor  # [B, C, T, H, W]
    # Permute to [B, T, C, H, W] for transformer
    return latents.permute(0, 2, 1, 3, 4)


# ---------------------------------------------------------------------------
# LR scheduler (linear warmup + constant)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    out_dir: str,
    step: int,
    tgnn: TGNNEncoder,
    adapter: GraphCrossAttnAdapter,
    optimizer: torch.optim.Optimizer,
) -> None:
    ckpt_dir = Path(out_dir) / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tgnn.state_dict(),    ckpt_dir / "tgnn.pt")
    torch.save(adapter.state_dict(), ckpt_dir / "adapter.pt")
    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    (Path(out_dir) / "latest_step.txt").write_text(str(step))
    print(f"  [ckpt] Saved to {ckpt_dir}")


def load_latest_checkpoint(
    out_dir: str,
    tgnn: TGNNEncoder,
    adapter: GraphCrossAttnAdapter,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    marker = Path(out_dir) / "latest_step.txt"
    if not marker.exists():
        return 0
    step = int(marker.read_text().strip())
    ckpt_dir = Path(out_dir) / f"step_{step:06d}"
    tgnn.load_state_dict(torch.load(ckpt_dir / "tgnn.pt",    map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(ckpt_dir / "adapter.pt", map_location=device, weights_only=True))
    optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location=device))
    print(f"  [ckpt] Resumed from step {step}")
    return step


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = cfg.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[cfg.mixed_precision]

    print(f"\n{'='*60}")
    print(f"  Device : {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"  Dtype  : {dtype}")
    print(f"{'='*60}\n")

    # ── Load frozen pipeline components ─────────────────────────────────────
    print("Loading CogVideoX pipeline components ...")
    pipe = CogVideoXPipeline.from_pretrained(
        cfg.pretrained_model,
        torch_dtype = dtype,
    )
    vae          = pipe.vae.to(device).eval()
    text_encoder = pipe.text_encoder.to(device).eval()
    tokenizer    = pipe.tokenizer
    transformer  = pipe.transformer.to(device).eval()

    for model in [vae, text_encoder, transformer]:
        model.requires_grad_(False)

    # VAE memory optimisations: slicing processes one video at a time in the
    # temporal loop; tiling applies spatial tiling for large spatial dimensions.
    vae.enable_slicing()
    vae.enable_tiling()

    # Enable gradient checkpointing on transformer to save memory during backward
    # (the adapter gradients must backprop through the transformer forward)
    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    # ── Noise scheduler ──────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(num_train_timesteps=cfg.n_timesteps)

    # ── Trainable modules ────────────────────────────────────────────────────
    print("Building TGNNEncoder + GraphCrossAttnAdapter ...")
    tgnn = TGNNEncoder(
        d_model  = cfg.d_model,
        n_layers = cfg.n_gnn_layers,
        n_heads  = cfg.n_gnn_heads,
        dropout  = cfg.dropout,
    ).to(device)

    # Cross-attention adapter: registers forward hooks on the frozen transformer.
    # graph_tokens are injected into each CogVideoXBlock via hook.
    adapter = GraphCrossAttnAdapter(
        transformer    = transformer,
        d_graph        = cfg.d_model,
        n_graph_tokens = cfg.n_graph_tokens,
        d_cross        = cfg.d_cross,
        n_heads        = cfg.n_cross_heads,
    ).to(device)

    trainable_params = list(tgnn.parameters()) + list(adapter.parameters())
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"  Trainable parameters: {n_trainable:,}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = cfg.lr,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    start_step = 0
    if cfg.resume:
        start_step = load_latest_checkpoint(cfg.out_dir, tgnn, adapter, optimizer, device)

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    print("Building dataset ...")
    dataset = AGTrainDataset(cfg)
    loader  = DataLoader(
        dataset,
        batch_size  = cfg.batch_size,
        shuffle     = True,
        num_workers = cfg.num_workers,
        collate_fn  = collate_fn,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,
    )

    # ── Mixed precision scaler ───────────────────────────────────────────────
    scaler = GradScaler(enabled=(cfg.mixed_precision == "fp16"))

    # ── Training loop ────────────────────────────────────────────────────────
    tgnn.train(); adapter.train()
    optimizer.zero_grad()

    step         = start_step
    accum_loss   = 0.0
    accum_count  = 0
    data_iter    = iter(loader)

    pbar = tqdm(total=cfg.n_steps - start_step, desc="Training")

    while step < cfg.n_steps:
        # Refill iterator at end of epoch
        try:
            graph_batch, frames, prompts = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            graph_batch, frames, prompts = next(data_iter)

        graph_batch = graph_batch.to(device)
        frames      = frames.to(device)             # [B, T, C, H, W]

        # ── 1. Encode frames → latents (no grad, frozen VAE) ─────────────
        latents = encode_frames(frames, vae, device, dtype)   # [B, T_lat, C, H, W]

        # ── 2. Sample noise + timestep ───────────────────────────────────
        B = latents.size(0)
        noise     = torch.randn_like(latents)
        timesteps = torch.randint(0, cfg.n_timesteps, (B,), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # ── 3. Encode text (no grad, frozen T5) ──────────────────────────
        text_emb = encode_prompt(
            prompts, tokenizer, text_encoder, device=device, dtype=dtype
        )  # [B, 226, 4096]

        # ── 4. Encode graph → node tokens → inject via cross-attn hooks ────
        with autocast(enabled=(cfg.mixed_precision != "no"), dtype=dtype):
            node_emb, _ = tgnn(graph_batch)                 # [N_total, d_model]
            graph_tokens = pad_node_embeddings(
                node_emb, graph_batch.batch,
                cfg.n_graph_tokens, cfg.d_model,
            ).to(dtype)                                      # [B, N_graph_tokens, d_model]
            adapter.set_graph_tokens(graph_tokens)

            # ── 5. Transformer forward (frozen) ───────────────────────────
            # Cross-attn hooks fire automatically inside each CogVideoXBlock.
            # encoder_hidden_states = text_emb only (226 tokens, no FiLM).
            pred_noise = transformer(
                hidden_states         = noisy_latents,
                encoder_hidden_states = text_emb,
                timestep              = timesteps,
                return_dict           = False,
            )[0]

            # ── 6. Loss ───────────────────────────────────────────────────
            loss = F.mse_loss(pred_noise.float(), noise.float())
            loss = loss / cfg.grad_accum

        # graph_tokens must stay set through backward: gradient checkpointing
        # recomputes the forward during backward, so hooks must still fire.
        scaler.scale(loss).backward()
        adapter.clear_graph_tokens()  # safe to clear only after backward
        accum_loss  += loss.item()
        accum_count += 1

        # Gradient accumulation step
        if accum_count == cfg.grad_accum:
            # Adjust LR (linear warmup)
            lr = get_lr(step, cfg.warmup_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % cfg.log_every == 0:
                print(
                    f"  step={step:6d}  loss={accum_loss:.5f}  "
                    f"lr={lr:.2e}  prompt='{prompts[0][:60]}'",
                    flush=True,
                )

            accum_loss  = 0.0
            accum_count = 0
            step       += 1
            pbar.update(1)

            if step % cfg.save_every == 0:
                save_checkpoint(cfg.out_dir, step, tgnn, adapter, optimizer)

    # Final checkpoint
    save_checkpoint(cfg.out_dir, step, tgnn, adapter, optimizer)
    pbar.close()
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> TrainConfig:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description="Train T-GNN adapter for CogVideoX")
    parser.add_argument("--out_dir",          default=cfg.out_dir)
    parser.add_argument("--graph_dir",        default=cfg.graph_dir)
    parser.add_argument("--frames_dir",       default=cfg.frames_dir)
    parser.add_argument("--pretrained_model", default=cfg.pretrained_model)
    parser.add_argument("--n_steps",          type=int,   default=cfg.n_steps)
    parser.add_argument("--batch_size",       type=int,   default=cfg.batch_size)
    parser.add_argument("--grad_accum",       type=int,   default=cfg.grad_accum)
    parser.add_argument("--lr",               type=float, default=cfg.lr)
    parser.add_argument("--frame_h",          type=int,   default=cfg.frame_h)
    parser.add_argument("--frame_w",          type=int,   default=cfg.frame_w)
    parser.add_argument("--n_frames",         type=int,   default=cfg.n_frames)
    parser.add_argument("--mixed_precision",  default=cfg.mixed_precision)
    parser.add_argument("--resume",           action="store_true")
    args = parser.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
