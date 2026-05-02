"""
train_wan.py
============
Stage 2: frozen Wan2.1-T2V-1.3B + trainable TGNNEncoder + WanGraphAdapter.

What is frozen  : VAE, UMT5 text encoder, Wan transformer (all parameters)
What is trained : TGNNEncoder (796K) + WanGraphAdapter (~4.5M)

Training objective (Flow Matching)
-----------------------------------
Unlike DDPM (ε-prediction), Wan uses Flow Matching:
    x_t  = (1 - σ) * x_0  +  σ * noise     (linear interpolation)
    v*   = noise - x_0                       (velocity target)
    v̂   = transformer(x_t, cond, σ)
    loss = MSE(v̂, v*)

σ is sampled uniformly from [0, 1]; the transformer timestep is σ × 1000 (int).

Resolution
----------
Wan2.1-1.3B supports 480P: 832×480 (landscape, 16:9).
This is the correct resolution — do NOT lower it to 256×256.

Frames: 33 (= 16×2+1 format for ~2 seconds at 16fps).
81 frames (full 5 sec) is too slow for training; 33 balances quality and speed.

Usage
-----
    python train_wan.py --out_dir /projects/.../checkpoints_wan
"""

from __future__ import annotations

import argparse, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from torch_geometric.data import Batch, Data
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model    import TGNNEncoder
from adapter_wan_v2 import WanGraphAdapterV2
from adapter        import pad_node_embeddings
from dataset_mixed  import MixedTrainDataset, MixedDataConfig, collate_fn_mixed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WanTrainConfig:
    pretrained_model: str  = "Wan-AI/Wan2.1-T2V-14B-Diffusers"   # 14B flagship
    graph_dir:        str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
    frames_dir:       str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
    charades_dir:     str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/videos/Charades_v1_480"
    out_dir:          str  = "/projects/bgnv/leatherman/tgnn/checkpoints_wan_14b"
    hf_cache:         str  = "/projects/bgnv/leatherman/hf_cache"

    split:       str = "train"
    n_frames:    int = 17      # 16×1+1 — shortest Wan format (~1s at 16fps)
    frame_h:     int = 480     # 480P native resolution
    frame_w:     int = 832     # 832×480 = 16:9 landscape
    num_workers: int = 4

    # 95/5 mixed training: 95% Charades text-only + 5% AG graph-conditioned
    graph_prob:  float = 0.05

    n_steps:        int   = 5_000
    batch_size:     int   = 1
    grad_accum:     int   = 4
    lr:             float = 1e-4
    weight_decay:   float = 0.01
    max_grad_norm:  float = 1.0
    warmup_steps:   int   = 200
    mixed_precision: str  = "bf16"   # Wan requires bf16

    d_model:        int = 256
    n_graph_tokens: int = 32
    n_gnn_layers:   int = 3
    n_gnn_heads:    int = 8
    dropout:        float = 0.1

    log_every:  int = 20
    save_every: int = 500
    resume:     bool = False


# ---------------------------------------------------------------------------
# Object class names (same as before)
# ---------------------------------------------------------------------------

_OBJECT_NAMES = [
    "__background__",
    "bag", "bed", "blanket", "book", "box", "broom", "chair", "closet",
    "clothes", "bottle", "dish", "door", "doorknob", "doorway",
    "floor", "food", "groceries", "laptop", "light", "medicine",
    "mirror", "notebook", "phone", "picture", "pillow",
    "refrigerator", "sandwich", "shelf", "shoe", "sofa",
    "table", "television", "towel", "vacuum", "window",
]


def graph_to_prompt(graph: Data) -> str:
    # Generic prompt: object names intentionally omitted.
    # Object information comes purely from graph conditioning.
    return "a person in an indoor scene"


# ---------------------------------------------------------------------------
# Dataset (same structure as AGTrainDataset but with Wan-compatible resolution)
# ---------------------------------------------------------------------------

class AGTrainDatasetWan(Dataset):
    def __init__(self, cfg: WanTrainConfig) -> None:
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
                    if (self.frames_dir / f"{g.video_id}.mp4").exists():
                        self.samples.append(pt)
            except Exception:
                pass
        print(f"  AGTrainDatasetWan ({cfg.split}): {len(self.samples)} videos")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Data, Tensor, str]:
        graph  = torch.load(self.samples[idx], weights_only=False)
        frames = self._load_frames(graph)
        prompt = graph_to_prompt(graph)
        return graph, frames, prompt

    def _load_frames(self, graph: Data) -> Tensor:
        folder   = self.frames_dir / f"{graph.video_id}.mp4"
        n        = self.cfg.n_frames
        C, H, W  = 3, self.cfg.frame_h, self.cfg.frame_w
        frame_ids = graph.frame_ids.tolist()
        if len(frame_ids) > n:
            step = len(frame_ids) / n
            frame_ids = [frame_ids[int(i * step)] for i in range(n)]
        frames_out: List[Tensor] = []
        last = torch.zeros(C, H, W)
        for fid in frame_ids:
            png = folder / f"{int(fid):06d}.png"
            if png.exists():
                last = self.transform(Image.open(png).convert("RGB"))
            frames_out.append(last)
        while len(frames_out) < n:
            frames_out.append(last)
        return torch.stack(frames_out[:n])   # [n, C, H, W]


def collate_fn(batch):
    graphs  = [item[0] for item in batch]
    frames  = torch.stack([item[1] for item in batch])
    prompts = [item[2] for item in batch]
    return Batch.from_data_list(graphs), frames, prompts


# ---------------------------------------------------------------------------
# Encode frames with Wan VAE
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_frames_wan(frames: Tensor, vae: AutoencoderKLWan, device, dtype) -> Tensor:
    """
    frames : [B, T, C, H, W]  float32 in [-1, 1]
    returns: [B, C_lat, T_lat, H_lat, W_lat]  normalized latents

    Wan VAE normalizes with latents_mean/latents_std (no single scaling_factor).
    Encode: normalized = (raw - mean) / std
    Decode (pipeline): raw = normalized * std + mean
    """
    # Wan VAE expects [B, C, T, H, W]
    x = frames.to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)
    with torch.autocast(device_type=device.type, dtype=dtype):
        posterior = vae.encode(x).latent_dist
    raw = posterior.sample()   # [B, C, T, H, W]

    # Normalize to zero-mean unit-variance space (same space the transformer was trained on)
    mean = torch.tensor(vae.config.latents_mean, device=device, dtype=raw.dtype).view(1, -1, 1, 1, 1)
    std  = torch.tensor(vae.config.latents_std,  device=device, dtype=raw.dtype).view(1, -1, 1, 1, 1)
    return (raw - mean) / std   # [B, C, T, H, W]


# ---------------------------------------------------------------------------
# Encode text with UMT5
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt_wan(
    prompts: List[str],
    tokenizer,
    text_encoder,
    max_length: int = 512,
    device=torch.device("cpu"),
    dtype=torch.float32,
) -> Tensor:
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids  = tokens.input_ids.to(device)
    mask = tokens.attention_mask.to(device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        emb = text_encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    return emb.to(dtype)   # [B, max_length, d_text]


# ---------------------------------------------------------------------------
# LR warmup
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup: int, base: float) -> float:
    return base * min(1.0, (step + 1) / warmup)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir, step, tgnn, adapter, optimizer):
    ckpt = Path(out_dir) / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(tgnn.state_dict(),      ckpt / "tgnn.pt")
    torch.save(adapter.state_dict(),   ckpt / "adapter.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    (Path(out_dir) / "latest_step.txt").write_text(str(step))
    print(f"  [ckpt] Saved → {ckpt}")


def load_checkpoint(out_dir, tgnn, adapter, optimizer, device) -> int:
    marker = Path(out_dir) / "latest_step.txt"
    if not marker.exists():
        return 0
    step = int(marker.read_text().strip())
    ckpt = Path(out_dir) / f"step_{step:06d}"
    tgnn.load_state_dict(torch.load(ckpt / "tgnn.pt",    map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(ckpt / "adapter.pt", map_location=device, weights_only=True))
    optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device))
    print(f"  [ckpt] Resumed from step {step}")
    return step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: WanTrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = cfg.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": torch.float32}[cfg.mixed_precision]
    print(f"\n{'='*60}")
    print(f"  Device : {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"  Dtype  : {dtype}")
    print(f"{'='*60}\n")

    # ── Load pipeline (frozen) ───────────────────────────────────────────────
    print("Loading Wan2.1 pipeline ...")
    pipe = WanPipeline.from_pretrained(cfg.pretrained_model, torch_dtype=dtype, local_files_only=True)
    vae          = pipe.vae.to(device).eval()
    text_encoder = pipe.text_encoder.to(device).eval()
    tokenizer    = pipe.tokenizer
    transformer  = pipe.transformer.to(device).eval()

    for m in [vae, text_encoder, transformer]:
        m.requires_grad_(False)

    vae.enable_slicing()
    vae.enable_tiling()

    if hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.pretrained_model, subfolder="scheduler", local_files_only=True
    )

    # ── Trainable modules ────────────────────────────────────────────────────
    print("Building TGNNEncoder + WanGraphAdapter ...")
    # text_dim in transformer config = T5 encoder output dim (4096 for UMT5-XXL)
    d_text = transformer.config.text_dim
    print(f"  T5 output dim (text_dim): {d_text}")

    tgnn = TGNNEncoder(
        d_model  = cfg.d_model,
        n_layers = cfg.n_gnn_layers,
        n_heads  = cfg.n_gnn_heads,
        dropout  = cfg.dropout,
    ).to(device)   # keep fp32 — GATv2Conv has bfloat16 compat issues

    d_inner = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    print(f"  Transformer inner_dim: {d_inner}")

    adapter = WanGraphAdapterV2(
        transformer    = transformer,
        d_graph        = cfg.d_model,
        n_graph_tokens = cfg.n_graph_tokens,
        d_inner        = d_inner,
    ).to(device=device, dtype=dtype)

    trainable_params = list(tgnn.parameters()) + list(adapter.parameters())
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8,
    )

    start_step = 0
    if cfg.resume:
        start_step = load_checkpoint(cfg.out_dir, tgnn, adapter, optimizer, device)

    # ── Dataset: 95% Charades text-only + 5% AG graph-conditioned ───────────
    print("Building mixed 95/5 dataset ...")
    mix_cfg = MixedDataConfig(
        charades_dir  = cfg.charades_dir,
        frames_dir    = cfg.frames_dir,
        graph_dir     = cfg.graph_dir,
        n_frames      = cfg.n_frames,
        frame_h       = cfg.frame_h,
        frame_w       = cfg.frame_w,
        graph_prob    = cfg.graph_prob,
        split         = cfg.split,
    )
    dataset = MixedTrainDataset(mix_cfg)
    loader  = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn_mixed,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )

    scaler    = GradScaler(enabled=(cfg.mixed_precision == "fp16"))
    tgnn.train(); adapter.train()
    optimizer.zero_grad()

    step        = start_step
    accum_loss  = 0.0
    accum_count = 0
    graph_steps = 0   # count graph-conditioned steps for logging
    data_iter   = iter(loader)
    pbar        = tqdm(total=cfg.n_steps - start_step, desc="Training")

    while step < cfg.n_steps:
        try:
            graph_batch, frames, prompts, has_graph = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            graph_batch, frames, prompts, has_graph = next(data_iter)

        frames = frames.to(device)  # [B, T, C, H, W]
        B      = frames.size(0)

        # ── 1. VAE encode: [B,T,C,H,W] → [B,C,T_lat,H_lat,W_lat] ──────
        latents = encode_frames_wan(frames, vae, device, dtype)

        # ── 2. Flow matching noise ───────────────────────────────────────
        # sigma ~ U[0,1]: noise level. x_t = (1-σ)*x_0 + σ*noise
        noise     = torch.randn_like(latents)
        sigma     = torch.rand(B, device=device, dtype=latents.dtype)
        s         = sigma.view(B, 1, 1, 1, 1)
        noisy     = (1.0 - s) * latents + s * noise
        v_target  = noise - latents      # velocity target: direction from data to noise
        timesteps = (sigma * 1000).long()  # transformer expects int timestep

        # ── 3. Text encode: prompts → [B, 512, 4096] ────────────────────
        text_emb = encode_prompt_wan(
            prompts, tokenizer, text_encoder, device=device, dtype=dtype,
        )

        # ── 4. Graph conditioning (only for 5% graph samples) ───────────
        with autocast(enabled=(cfg.mixed_precision != "no"), dtype=dtype):
            if has_graph and graph_batch is not None:
                # Graph-conditioned path (5% of steps):
                # TGNNEncoder → node_emb [N_total, 256] → pad → [B, 32, 256]
                # Adapter projects 256→d_inner and hook injects into encoder_hs
                graph_batch = graph_batch.to(device)
                node_emb, _ = tgnn(graph_batch)
                # node_tokens: [B, n_graph_tokens, d_model]
                node_tokens = pad_node_embeddings(
                    node_emb, graph_batch.batch, cfg.n_graph_tokens, cfg.d_model,
                ).to(dtype)
                adapter.set_graph_tokens(node_tokens)
                graph_steps += 1
            else:
                # Text-only path (95% of steps):
                # No graph tokens → hook fires but injects zeros → pure text generation
                adapter.clear_graph_tokens()

            # ── 5. Transformer forward ──────────────────────────────────
            # Hook inside condition_embedder injects graph tokens (if set)
            # into encoder_hs [B, 512→544, d_inner] before cross-attention.
            # v_pred: [B, C, T_lat, H_lat, W_lat] — predicted velocity field
            v_pred = transformer(
                hidden_states         = noisy,      # [B, C, T_lat, H_lat, W_lat]
                encoder_hidden_states = text_emb,   # [B, 512, 4096]
                timestep              = timesteps,   # [B] int in [0, 1000]
                return_dict           = False,
            )[0]

            # ── 6. Flow matching loss ────────────────────────────────────
            # MSE between predicted and target velocity field
            loss = F.mse_loss(v_pred.float(), v_target.float()) / cfg.grad_accum

        # Keep graph_tokens set through backward: gradient checkpointing
        # recomputes condition_embedder during backward, hook must still fire.
        scaler.scale(loss).backward()
        adapter.clear_graph_tokens()   # safe to clear only after backward
        accum_loss  += loss.item()
        accum_count += 1

        if accum_count == cfg.grad_accum:
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
                    f"lr={lr:.2e}  graph_steps={graph_steps}",
                    flush=True,
                )

            accum_loss  = 0.0
            accum_count = 0
            step       += 1
            pbar.update(1)

            if step % cfg.save_every == 0:
                save_checkpoint(cfg.out_dir, step, tgnn, adapter, optimizer)

    save_checkpoint(cfg.out_dir, step, tgnn, adapter, optimizer)
    pbar.close()
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    cfg = WanTrainConfig()
    p   = argparse.ArgumentParser()
    p.add_argument("--out_dir",          default=cfg.out_dir)
    p.add_argument("--n_steps",          type=int,   default=cfg.n_steps)
    p.add_argument("--batch_size",       type=int,   default=cfg.batch_size)
    p.add_argument("--grad_accum",       type=int,   default=cfg.grad_accum)
    p.add_argument("--lr",               type=float, default=cfg.lr)
    p.add_argument("--frame_h",          type=int,   default=cfg.frame_h)
    p.add_argument("--frame_w",          type=int,   default=cfg.frame_w)
    p.add_argument("--n_frames",         type=int,   default=cfg.n_frames)
    p.add_argument("--mixed_precision",  default=cfg.mixed_precision)
    p.add_argument("--resume",           action="store_true")
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    train(parse_args())
