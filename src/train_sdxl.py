"""
train_sdxl.py
=============
SDXL Base + TGNNEncoder + SDXLGraphAdapter training.

Frozen:   VAE, text_encoder (CLIP-L), text_encoder_2 (CLIP-G), UNet base weights
Trained:  TGNNEncoder (~800K), SDXLGraphAdapter (~25M)

Loss: standard SDXL DDIM/DDPM noise prediction (epsilon objective).

Validation
----------
Every `val_every` optimizer steps:
  - Compute MSE noise prediction loss on `val_batches` graph-cond samples
  - Track best val_loss; save 'best.pt' checkpoint
  - Early-stop if no improvement for `val_patience` evaluations

This gives us reliable in-training feedback we never had with HunyuanVideo.
"""

from __future__ import annotations

import argparse, os, sys, json, time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
)

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model    import TGNNEncoder
from adapter_sdxl  import SDXLGraphAdapter, pad_node_embeddings_temporal
from dataset_image import (
    ImageDataConfig, MixedImageDataset, AGImageDataset, collate_fn_image,
)

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SDXLTrainConfig:
    model_id:     str = MODEL_ID
    out_dir:      str = "/projects/bgnv/leatherman/tgnn/checkpoints_sdxl"
    hf_cache:     str = "/projects/bgnv/leatherman/hf_cache"

    # Data
    image_h:     int   = 1024
    image_w:     int   = 1024
    graph_prob:  float = 0.30   # 70/30 mix
    num_workers: int   = 4

    # Train
    n_steps:        int   = 30_000
    batch_size:     int   = 2     # SDXL UNet @1024 fits 2 on H200 80GB
    grad_accum:     int   = 4
    lr:             float = 1e-4
    weight_decay:   float = 0.01
    max_grad_norm:  float = 1.0
    warmup_steps:   int   = 500

    # Adapter
    d_model:        int = 256
    n_graph_tokens: int = 64

    # Init class_embed from CLIP names
    use_clip_class_init: bool = True

    # Validation / early stopping
    val_every:    int = 1000
    val_batches:  int = 32
    val_patience: int = 5      # if no improvement in 5 evals → early stop
    save_every:   int = 2000

    # Logging
    log_every: int  = 50
    resume:    bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_sdxl(images: Tensor, vae, dtype) -> Tensor:
    """[B,3,H,W] in [-1,1] → latents [B,4,H/8,W/8]

    SDXL VAE has documented fp16 instability (overflows → NaN). We force
    fp32 for the VAE pass and cast latents back to the model dtype.
    """
    x = images.to(dtype=torch.float32)
    posterior = vae.encode(x).latent_dist
    latents = posterior.sample() * vae.config.scaling_factor
    return latents.to(dtype)


@torch.no_grad()
def encode_prompt_sdxl(pipe, prompts: List[str], device, dtype):
    """
    Returns
        prompt_embeds : [B, 77, 2048]   (CLIP-L 768 + CLIP-G 1280 concat)
        pooled        : [B, 1280]
    """
    out = pipe.encode_prompt(prompt=prompts, device=device, do_classifier_free_guidance=False)
    prompt_embeds, _, pooled, _ = out
    return prompt_embeds.to(dtype), pooled.to(dtype)


def get_lr(step, warmup, base_lr):
    return base_lr * min(1.0, (step + 1) / max(1, warmup))


def save_ckpt(out_dir, tag, tgnn, adapter, optimizer, step):
    ckpt = Path(out_dir) / tag
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(tgnn.state_dict(),       ckpt / "tgnn.pt")
    torch.save(adapter.state_dict(),    ckpt / "adapter.pt")
    torch.save(optimizer.state_dict(),  ckpt / "optimizer.pt")
    (ckpt / "step.txt").write_text(str(step))
    (Path(out_dir) / "latest_tag.txt").write_text(tag)
    print(f"  [ckpt] saved → {ckpt}", flush=True)


def load_ckpt(out_dir, tgnn, adapter, optimizer, device) -> int:
    marker = Path(out_dir) / "latest_tag.txt"
    if not marker.exists():
        return 0
    tag = marker.read_text().strip()
    ckpt = Path(out_dir) / tag
    tgnn.load_state_dict(torch.load(ckpt / "tgnn.pt",       map_location=device, weights_only=True))
    adapter.load_state_dict(torch.load(ckpt / "adapter.pt", map_location=device, weights_only=True))
    optimizer.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location=device))
    step = int((ckpt / "step.txt").read_text().strip())
    print(f"  [ckpt] resumed from {tag} @ step {step}")
    return step


# ---------------------------------------------------------------------------
# SDXL "added time IDs" (required for SDXL UNet conditioning)
# ---------------------------------------------------------------------------

def get_add_time_ids(image_h, image_w, dtype, batch_size, device):
    """
    SDXL UNet expects added_time_ids = [original_h, original_w, crop_top, crop_left,
                                         target_h, target_w] for each batch element.
    """
    ids = torch.tensor(
        [image_h, image_w, 0, 0, image_h, image_w],
        dtype=dtype, device=device,
    )
    return ids.unsqueeze(0).expand(batch_size, -1)


# ---------------------------------------------------------------------------
# Main forward step (used by both train & val)
# ---------------------------------------------------------------------------

def forward_step(
    cfg, batch, pipe, tgnn, adapter, scheduler,
    device, dtype, training: bool,
):
    """
    Returns (loss, n_graph_in_batch).
    """
    graph_batch, images, target_idxs, prompts, has_graph_mask = batch
    images       = images.to(device)
    target_idxs  = target_idxs.to(device)
    has_graph_mask = has_graph_mask.to(device)
    B = images.size(0)

    latents = encode_image_sdxl(images, pipe.vae, dtype)             # [B,4,h,w]

    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps,
                              (B,), device=device, dtype=torch.long)
    noisy = scheduler.add_noise(latents, noise, timesteps)

    prompt_embeds, pooled = encode_prompt_sdxl(pipe, prompts, device, dtype)

    # Graph tokens: real for graph items, zeros for text-only items
    if graph_batch is not None:
        graph_batch = graph_batch.to(device)
        # Slice target_idxs for graph items only
        graph_target_idxs = target_idxs[has_graph_mask]              # [B_g]
        node_emb, _, is_target_node = tgnn(graph_batch, graph_target_idxs)
        node_tokens_g = pad_node_embeddings_temporal(
            node_emb=node_emb,
            batch_idx=graph_batch.batch,
            node_frame=graph_batch.node_frame,
            target_frame_idx=graph_target_idxs,
            is_target_node=is_target_node,
            n_tokens=cfg.n_graph_tokens,
            d_graph=cfg.d_model,
        ).to(dtype)                                                  # [B_g, N, 256]
        # Place back into B-sized tensor (text-only rows = zero)
        node_tokens = torch.zeros(B, cfg.n_graph_tokens, cfg.d_model,
                                  device=device, dtype=dtype)
        node_tokens[has_graph_mask] = node_tokens_g
    else:
        node_tokens = torch.zeros(B, cfg.n_graph_tokens, cfg.d_model,
                                  device=device, dtype=dtype)

    graph_tokens = adapter.encode(node_tokens)                       # [B, N, 2048]
    adapter.set_graph_tokens(graph_tokens)

    add_time_ids = get_add_time_ids(cfg.image_h, cfg.image_w, dtype, B, device)
    added_cond_kwargs = {"text_embeds": pooled, "time_ids": add_time_ids}

    noise_pred = pipe.unet(
        noisy, timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]
    adapter.clear_graph_tokens()

    loss = F.mse_loss(noise_pred.float(), noise.float())
    return loss, int(has_graph_mask.sum().item())


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(cfg, val_loader, pipe, tgnn, adapter, scheduler, device, dtype) -> float:
    tgnn.eval(); adapter.eval()
    total = 0.0; n = 0
    for i, batch in enumerate(val_loader):
        if i >= cfg.val_batches: break
        loss, _ = forward_step(cfg, batch, pipe, tgnn, adapter, scheduler,
                               device, dtype, training=False)
        total += loss.item(); n += 1
    tgnn.train(); adapter.train()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: SDXLTrainConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.environ["HF_HOME"] = cfg.hf_cache
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 over fp16: fp16 overflows in SDXL attention (NaN by ~step 50).
    # bf16 has fp32-equivalent exponent range, no GradScaler needed.
    dtype  = torch.bfloat16

    print(f"\n{'='*60}\n  Device: {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})\n{'='*60}\n")

    # ── Pipeline (frozen) ───────────────────────────────────────────────────
    print("Loading SDXL pipeline ...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        cfg.model_id, torch_dtype=dtype, local_files_only=True,
    ).to(device)
    pipe.vae.to(torch.float32)                                    # fp16 VAE = NaN
    pipe.vae.eval(); pipe.text_encoder.eval(); pipe.text_encoder_2.eval()
    pipe.unet.eval()                                              # base weights frozen
    for m in [pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.unet]:
        m.requires_grad_(False)

    scheduler = DDPMScheduler.from_pretrained(
        cfg.model_id, subfolder="scheduler", local_files_only=True,
    )

    cross_dim = pipe.unet.config.cross_attention_dim                # 2048 for SDXL

    # ── Trainable modules ──────────────────────────────────────────────────
    print("Building TGNN + adapter ...")
    tgnn = TGNNEncoder(d_model=cfg.d_model).to(device)               # fp32 for GATv2

    if cfg.use_clip_class_init:
        ag_classes_path = (
            "/projects/bgnv/leatherman/tgnn/dataset/ag/annotations/"
            "action_genome_v1.0/object_classes.txt"
        )
        ag_classes = [l.strip() for l in open(ag_classes_path) if l.strip()]
        print(f"  Initializing class_embed from CLIP for {len(ag_classes)} classes ...")
        tgnn.init_class_embed_from_clip(
            ag_classes, clip_model_name="openai/clip-vit-large-patch14",
        )

    adapter = SDXLGraphAdapter(d_graph=cfg.d_model, cross_dim=cross_dim).to(device, dtype)
    adapter.install(pipe.unet)
    # The newly-added attn processors hold parameters in fp16 due to .to(dtype)
    adapter.projector.to(dtype)

    trainable = list(tgnn.parameters()) + list(adapter.parameters())
    n_params = sum(p.numel() for p in trainable)
    print(f"  Trainable params: TGNN={sum(p.numel() for p in tgnn.parameters()):,} + "
          f"adapter={adapter.num_trainable_params():,} = {n_params:,}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_step = load_ckpt(cfg.out_dir, tgnn, adapter, optimizer, device) if cfg.resume else 0

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\nBuilding datasets ...")
    train_cfg = ImageDataConfig(image_h=cfg.image_h, image_w=cfg.image_w,
                                graph_prob=cfg.graph_prob, split="train")
    val_cfg   = ImageDataConfig(image_h=cfg.image_h, image_w=cfg.image_w,
                                graph_prob=1.0, split="val")
    train_ds = MixedImageDataset(train_cfg)
    val_ds   = AGImageDataset(val_cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_fn_image,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                              num_workers=2, collate_fn=collate_fn_image, drop_last=True)

    tgnn.train(); adapter.train()
    optimizer.zero_grad()

    # ── Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "step": []}
    best_val = float("inf"); val_strikes = 0
    step = start_step
    accum_loss = 0.0; accum_count = 0; graph_seen = 0
    data_iter = iter(train_loader)
    pbar = tqdm(total=cfg.n_steps - start_step, desc="train")

    while step < cfg.n_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader); batch = next(data_iter)

        loss, ng = forward_step(cfg, batch, pipe, tgnn, adapter, scheduler,
                                device, dtype, training=True)
        loss = loss / cfg.grad_accum
        loss.backward()
        accum_loss  += loss.item()
        accum_count += 1
        graph_seen  += ng

        if accum_count == cfg.grad_accum:
            lr = get_lr(step, cfg.warmup_steps, cfg.lr)
            for pg in optimizer.param_groups: pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
            optimizer.step(); optimizer.zero_grad()

            if step % cfg.log_every == 0:
                history["train_loss"].append(accum_loss); history["step"].append(step)
                print(f"  step={step:6d}  loss={accum_loss:.5f}  "
                      f"lr={lr:.2e}  graph_seen={graph_seen}", flush=True)

            accum_loss = 0.0; accum_count = 0
            step += 1; pbar.update(1)

            # ── Validation ─────────────────────────────────────────
            if step % cfg.val_every == 0:
                t0 = time.time()
                vloss = validate(cfg, val_loader, pipe, tgnn, adapter,
                                 scheduler, device, dtype)
                history["val_loss"].append((step, vloss))
                print(f"  ── VAL @ step {step}: loss={vloss:.5f}  "
                      f"(took {time.time()-t0:.1f}s)", flush=True)
                if vloss < best_val - 1e-4:
                    best_val = vloss; val_strikes = 0
                    save_ckpt(cfg.out_dir, "best", tgnn, adapter, optimizer, step)
                else:
                    val_strikes += 1
                    print(f"  no improvement ({val_strikes}/{cfg.val_patience})")
                    if val_strikes >= cfg.val_patience:
                        print("  ⚠ early stop"); break

            if step % cfg.save_every == 0:
                save_ckpt(cfg.out_dir, f"step_{step:06d}", tgnn, adapter, optimizer, step)

    save_ckpt(cfg.out_dir, f"step_{step:06d}", tgnn, adapter, optimizer, step)
    (Path(cfg.out_dir) / "history.json").write_text(json.dumps(history, indent=2))
    pbar.close()
    print(f"\nTraining complete. best_val={best_val:.5f}")


def parse_args():
    cfg = SDXLTrainConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",  default=cfg.out_dir)
    p.add_argument("--n_steps",  type=int,   default=cfg.n_steps)
    p.add_argument("--lr",       type=float, default=cfg.lr)
    p.add_argument("--graph_prob", type=float, default=cfg.graph_prob)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--resume",   action="store_true")
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    train(parse_args())
