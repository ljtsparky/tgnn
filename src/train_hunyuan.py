"""
train_hunyuan.py
================
Stage 2: frozen HunyuanVideo + trainable TGNNEncoder + HunyuanGraphAdapter.

Frozen:   VAE, LLaMA text encoder, CLIP text encoder, HunyuanVideo transformer
Trained:  TGNNEncoder (796K params) + HunyuanGraphAdapter (~4.5M params) = ~5.3M

Training objective: Flow Matching (same as Wan2.1)
    x_t  = (1-σ)*x_0 + σ*noise     sigma ~ U[0,1]
    v*   = noise - x_0              velocity target
    v̂   = transformer(x_t, cond, σ)
    loss = MSE(v̂, v*)

Data: 95% Charades text-only + 5% Action Genome graph-conditioned (95/5 mix)
    Text-only samples: no graph tokens injected, trains model not to degrade
    Graph samples: graph_tokens prepended to LLaMA sequence, trains adapter

GPU: H200 (80GB) recommended. 1 H200 sufficient for training with sequential offload.
"""

from __future__ import annotations

import argparse, os, sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch_geometric.data import Batch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model      import TGNNEncoder
from adapter_hunyuan import HunyuanGraphAdapter
from adapter         import pad_node_embeddings
from dataset_mixed   import MixedTrainDataset, MixedDataConfig, collate_fn_mixed

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HunyuanTrainConfig:
    model_id:     str  = MODEL_ID
    graph_dir:    str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
    frames_dir:   str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
    charades_dir: str  = "/projects/bgnv/leatherman/tgnn/dataset/ag/videos/Charades_v1_480"
    out_dir:      str  = "/projects/bgnv/leatherman/tgnn/checkpoints_hunyuan"
    hf_cache:     str  = "/projects/bgnv/leatherman/hf_cache"

    # Frames — 16×1+1 format, 1 second at 16fps
    split:      str   = "train"
    n_frames:   int   = 17
    frame_h:    int   = 480
    frame_w:    int   = 848
    num_workers: int  = 4

    # 95/5 mixed training
    graph_prob:  float = 0.05

    # Training
    n_steps:        int   = 5_000
    batch_size:     int   = 1
    grad_accum:     int   = 4
    lr:             float = 1e-4
    weight_decay:   float = 0.01
    max_grad_norm:  float = 1.0
    warmup_steps:   int   = 200

    # Adapter dims
    d_model:        int   = 256
    n_graph_tokens: int   = 32

    # Logging
    log_every:  int  = 20
    save_every: int  = 500
    resume:     bool = False


# ---------------------------------------------------------------------------
# VAE encode for HunyuanVideo
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_frames_hunyuan(frames: Tensor, vae, device, dtype) -> Tensor:
    """
    frames : [B, T, C, H, W]  float32 in [-1, 1]
    returns: [B, C, T_lat, H_lat, W_lat]  normalized latents

    HunyuanVideo VAE:
      - spatial: 8× compression  (480→60, 848→106)
      - temporal: 4× compression (17→5 frames for 17-frame input)
      - scaling_factor: 0.476986
    """
    # HunyuanVideo VAE expects [B, C, T, H, W]
    x = frames.to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)
    with torch.autocast(device_type=device.type, dtype=dtype):
        posterior = vae.encode(x).latent_dist
    # HunyuanVideo uses scaling_factor directly (no per-channel mean/std)
    return posterior.sample() * vae.config.scaling_factor  # [B, C, T_lat, H_lat, W_lat]


# ---------------------------------------------------------------------------
# Text encode for HunyuanVideo (LLaMA + CLIP)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_prompt_hunyuan(pipe, prompts: List[str], device, max_length: int = 256):
    """
    Returns:
        prompt_embeds      : [B, seq, 4096]  — LLaMA sequence
        pooled_embeds      : [B, 768]        — CLIP pooled
        attention_mask     : [B, seq]        — LLaMA attention mask
    """
    prompt_embeds_list = []
    pooled_list        = []
    mask_list          = []

    for prompt in prompts:
        # encode_prompt returns (prompt_embeds, pooled_prompt_embeds, prompt_attention_mask)
        embeds, pooled, mask = pipe.encode_prompt(
            prompt=prompt,
            max_sequence_length=max_length,
            device=device,
        )
        prompt_embeds_list.append(embeds)
        pooled_list.append(pooled)
        mask_list.append(mask)

    return (
        torch.cat(prompt_embeds_list, dim=0),   # [B, seq, 4096]
        torch.cat(pooled_list,        dim=0),   # [B, 768]
        torch.cat(mask_list,          dim=0),   # [B, seq]
    )


# ---------------------------------------------------------------------------
# LR + checkpoint helpers
# ---------------------------------------------------------------------------

def get_lr(step, warmup, base_lr):
    return base_lr * min(1.0, (step + 1) / warmup)


def save_ckpt(out_dir, step, tgnn, adapter, optimizer):
    ckpt = Path(out_dir) / f"step_{step:06d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(tgnn.state_dict(),      ckpt / "tgnn.pt")
    torch.save(adapter.state_dict(),   ckpt / "adapter.pt")
    torch.save(optimizer.state_dict(), ckpt / "optimizer.pt")
    (Path(out_dir) / "latest_step.txt").write_text(str(step))
    print(f"  [ckpt] Saved → {ckpt}")


def load_ckpt(out_dir, tgnn, adapter, optimizer, device) -> int:
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

def train(cfg: HunyuanTrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = cfg.hf_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # HunyuanVideo transformer uses bf16; rest uses fp16
    dtype_transformer = torch.bfloat16
    dtype_vae         = torch.float16
    print(f"\n{'='*60}")
    print(f"  Device : {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")
    print(f"{'='*60}\n")

    # ── Load pipeline (all frozen) ───────────────────────────────────────────
    print("Loading HunyuanVideo pipeline ...")
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        cfg.model_id, subfolder="transformer",
        torch_dtype=dtype_transformer, local_files_only=True,
    )
    pipe = HunyuanVideoPipeline.from_pretrained(
        cfg.model_id, transformer=transformer,
        torch_dtype=dtype_vae, local_files_only=True,
    )
    # Move components to GPU; offload text encoders to CPU after encoding
    pipe.transformer.to(device)
    pipe.vae.to(device).eval()
    pipe.text_encoder.to(device).eval()
    pipe.text_encoder_2.to(device).eval()
    pipe.vae.enable_tiling()

    for m in [pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]:
        m.requires_grad_(False)

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model_id, subfolder="scheduler", local_files_only=True,
    )

    # ── Trainable modules ────────────────────────────────────────────────────
    print("Building TGNNEncoder + HunyuanGraphAdapter ...")
    tgnn    = TGNNEncoder(d_model=cfg.d_model).to(device)   # fp32 (GATv2Conv compat)
    adapter = HunyuanGraphAdapter(
        d_graph=cfg.d_model, d_text=4096, n_graph_tokens=cfg.n_graph_tokens,
    ).to(device=device, dtype=dtype_transformer)

    trainable = list(tgnn.parameters()) + list(adapter.parameters())
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_step = load_ckpt(cfg.out_dir, tgnn, adapter, optimizer, device) if cfg.resume else 0

    # ── Mixed dataset (95% Charades + 5% AG) ────────────────────────────────
    print("Building mixed 95/5 dataset ...")
    mix_cfg = MixedDataConfig(
        charades_dir=cfg.charades_dir, frames_dir=cfg.frames_dir,
        graph_dir=cfg.graph_dir, n_frames=cfg.n_frames,
        frame_h=cfg.frame_h, frame_w=cfg.frame_w,
        graph_prob=cfg.graph_prob, split=cfg.split,
    )
    dataset = MixedTrainDataset(mix_cfg)
    loader  = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate_fn_mixed,
        pin_memory=(device.type == "cuda"), drop_last=True,
    )

    scaler = GradScaler(enabled=False)   # bf16 doesn't use GradScaler
    tgnn.train(); adapter.train()
    optimizer.zero_grad()

    step        = start_step
    accum_loss  = 0.0
    accum_count = 0
    graph_steps = 0
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

        # ── 1. VAE encode → latents [B, C, T_lat, H_lat, W_lat] ────────
        latents = encode_frames_hunyuan(frames, pipe.vae, device, dtype_vae).to(dtype_transformer)

        # ── 2. Flow matching noise ───────────────────────────────────────
        noise     = torch.randn_like(latents)
        sigma     = torch.rand(B, device=device, dtype=dtype_transformer)
        s         = sigma.view(B, 1, 1, 1, 1)
        noisy     = (1.0 - s) * latents + s * noise   # [B, C, T_lat, H_lat, W_lat]
        v_target  = noise - latents
        # HunyuanVideo timestep must be float in [0, 1000] (not long like DDPM)
        timesteps = (sigma * 1000).to(dtype_transformer)   # [B] float
        # Embedded guidance: required when transformer.config.guidance_embeds=True
        guidance  = torch.full((B,), 6.0, device=device, dtype=dtype_transformer)

        # ── 3. Text encode (move encoders to CPU after use to save memory) ─
        prompt_embeds, pooled_embeds, attn_mask = encode_prompt_hunyuan(
            pipe, prompts, device,
        )
        prompt_embeds = prompt_embeds.to(dtype_transformer)   # [B, seq, 4096]
        pooled_embeds = pooled_embeds.to(dtype_transformer)   # [B, 768]

        # ── 4. Graph conditioning (5%) or text-only (95%) ───────────────
        # IMPORTANT: we ALWAYS call adapter to maintain gradient path.
        # For text-only samples we pass zero node_tokens so the adapter
        # contributes ~0 to combined_embeds (zero-init weights + zero input = 0).
        # This ensures loss.backward() has a valid gradient graph in both paths.
        if has_graph and graph_batch is not None:
            # Graph-conditioned path: real node embeddings
            graph_batch = graph_batch.to(device)
            node_emb, _ = tgnn(graph_batch)           # [N_total, 256] fp32
            node_tokens = pad_node_embeddings(
                node_emb, graph_batch.batch, cfg.n_graph_tokens, cfg.d_model,
            ).to(dtype_transformer)                   # [B, N, 256]
            graph_steps += 1
        else:
            # Text-only path: zero node tokens → adapter injects ~0 (no graph signal)
            node_tokens = torch.zeros(
                B, cfg.n_graph_tokens, cfg.d_model,
                device=device, dtype=dtype_transformer,
            )
        # Always project through adapter → gradient path guaranteed
        combined_embeds, combined_mask = adapter(node_tokens, prompt_embeds, attn_mask)

        # ── 5. Transformer forward ───────────────────────────────────────
        # HunyuanVideo transformer takes:
        #   hidden_states:          [B, C, T_lat, H_lat, W_lat] latents
        #   encoder_hidden_states:  [B, seq, 4096] text/graph tokens
        #   pooled_projections:     [B, 768] CLIP pooled
        #   timestep:               [B] timestep indices
        #   encoder_attention_mask: [B, seq] attention mask
        v_pred = pipe.transformer(
            hidden_states           = noisy,
            encoder_hidden_states   = combined_embeds,
            pooled_projections      = pooled_embeds,
            timestep                = timesteps,
            encoder_attention_mask  = combined_mask,
            guidance                = guidance,  # required: guidance_embeds=True
            return_dict             = False,
        )[0]

        # ── 6. Flow matching loss ────────────────────────────────────────
        loss = F.mse_loss(v_pred.float(), v_target.float()) / cfg.grad_accum
        loss.backward()
        accum_loss  += loss.item()
        accum_count += 1

        if accum_count == cfg.grad_accum:
            lr = get_lr(step, cfg.warmup_steps, cfg.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            if step % cfg.log_every == 0:
                print(f"  step={step:6d}  loss={accum_loss:.5f}  "
                      f"lr={lr:.2e}  graph_steps={graph_steps}", flush=True)

            accum_loss  = 0.0
            accum_count = 0
            step       += 1
            pbar.update(1)

            if step % cfg.save_every == 0:
                save_ckpt(cfg.out_dir, step, tgnn, adapter, optimizer)

    save_ckpt(cfg.out_dir, step, tgnn, adapter, optimizer)
    pbar.close()
    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    cfg = HunyuanTrainConfig()
    p   = argparse.ArgumentParser()
    p.add_argument("--out_dir",   default=cfg.out_dir)
    p.add_argument("--n_steps",   type=int,   default=cfg.n_steps)
    p.add_argument("--lr",        type=float, default=cfg.lr)
    p.add_argument("--resume",    action="store_true")
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    train(parse_args())
