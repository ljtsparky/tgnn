"""
test_hunyuan_dryrun.py
======================
Dry-run: verify all shapes for HunyuanVideo + TGNNEncoder + HunyuanGraphAdapter.
No backward pass. Checks: pipeline load, VAE encode, flow matching,
text encode, graph encode + inject, transformer forward, loss.
"""

import sys, os, torch, torch.nn.functional as F
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model      import TGNNEncoder
from adapter_hunyuan import HunyuanGraphAdapter
from adapter         import pad_node_embeddings
from train_hunyuan   import (HunyuanTrainConfig, encode_frames_hunyuan,
                              encode_prompt_hunyuan)
from dataset_mixed   import MixedDataConfig, MixedTrainDataset, collate_fn_mixed

HF_CACHE = "/projects/bgnv/leatherman/hf_cache"
MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
N_GRAPH_TOKENS = 32

def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond: sys.exit(1)

os.environ["HF_HOME"] = HF_CACHE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype_t = torch.bfloat16
dtype_v = torch.float16
print(f"\nDevice: {device}  transformer={dtype_t}  vae={dtype_v}")

# ── 1. Load pipeline ──────────────────────────────────────────────────────
print("\n=== 1. Load HunyuanVideo pipeline ===")
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    MODEL_ID, subfolder="transformer", torch_dtype=dtype_t, local_files_only=True,
)
pipe = HunyuanVideoPipeline.from_pretrained(
    MODEL_ID, transformer=transformer, torch_dtype=dtype_v, local_files_only=True,
)
pipe.transformer.to(device); pipe.vae.to(device).eval()
pipe.text_encoder.to(device).eval(); pipe.text_encoder_2.to(device).eval()
pipe.vae.enable_tiling()
n_double = len(pipe.transformer.transformer_blocks)
n_single = len(pipe.transformer.single_transformer_blocks)
d_inner  = pipe.transformer.config.num_attention_heads * pipe.transformer.config.attention_head_dim
print(f"  Transformer: {n_double} double + {n_single} single blocks, d_inner={d_inner}")
print(f"  VAE scaling_factor: {pipe.vae.config.scaling_factor}")

# ── 2. Load dataset sample ────────────────────────────────────────────────
print("\n=== 2. Load graph + frames (480×848, 17 frames) ===")
cfg = HunyuanTrainConfig()
cfg.n_frames = 17; cfg.frame_h = 480; cfg.frame_w = 848
mix = MixedDataConfig(charades_dir=cfg.charades_dir, frames_dir=cfg.frames_dir,
                      graph_dir=cfg.graph_dir, n_frames=17, frame_h=480, frame_w=848,
                      graph_prob=1.0)  # always graph for dryrun
ds = MixedTrainDataset(mix)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=collate_fn_mixed, drop_last=True)
graph_batch_raw, frames, prompts, has_graph = next(iter(loader))
graph_batch = graph_batch_raw.to(device)
frames      = frames.to(device)
B = frames.size(0)
print(f"  frames: {frames.shape}  has_graph={has_graph}")
check(frames.shape == (1, 17, 3, 480, 848), "frames [1,17,3,480,848]")

# ── 3. VAE encode ─────────────────────────────────────────────────────────
print("\n=== 3. VAE encode ===")
latents = encode_frames_hunyuan(frames, pipe.vae, device, dtype_v).to(dtype_t)
print(f"  latents: {latents.shape}")  # [B, 16, T_lat, H_lat, W_lat]
check(latents.ndim == 5, "latents 5-D")

# ── 4. Flow matching noise ────────────────────────────────────────────────
print("\n=== 4. Flow matching noise ===")
noise     = torch.randn_like(latents)
sigma     = torch.rand(B, device=device, dtype=dtype_t)
noisy     = (1-sigma.view(B,1,1,1,1))*latents + sigma.view(B,1,1,1,1)*noise
v_target  = noise - latents
# HunyuanVideo timestep: float in [0, 1000], NOT long (unlike DDPM)
timesteps = (sigma * 1000).to(dtype_t)   # [B] float bfloat16
# guidance tensor required when guidance_embeds=True (embedded CFG)
guidance  = torch.full((B,), 6.0, device=device, dtype=dtype_t)
print(f"  noisy: {noisy.shape}  timesteps: {timesteps}")
check(noisy.shape == latents.shape, "noisy shape matches latents")

# ── 5. Text encode ────────────────────────────────────────────────────────
print("\n=== 5. Text encode ===")
prompt_embeds, pooled_embeds, attn_mask = encode_prompt_hunyuan(pipe, prompts, device)
prompt_embeds = prompt_embeds.to(dtype_t)
pooled_embeds = pooled_embeds.to(dtype_t)
print(f"  prompt_embeds: {prompt_embeds.shape}")   # [1, seq, 4096]
print(f"  pooled_embeds: {pooled_embeds.shape}")   # [1, 768]
check(prompt_embeds.shape[-1] == 4096, "text dim = 4096")

# ── 6. TGNNEncoder + HunyuanGraphAdapter ─────────────────────────────────
print("\n=== 6. TGNNEncoder + HunyuanGraphAdapter ===")
tgnn    = TGNNEncoder(d_model=256).to(device).eval()
adapter = HunyuanGraphAdapter(d_graph=256, d_text=4096, n_graph_tokens=N_GRAPH_TOKENS
                              ).to(device=device, dtype=dtype_t).eval()
n_params = sum(p.numel() for p in list(tgnn.parameters()) + list(adapter.parameters()))
print(f"  Total trainable params: {n_params:,}")

with torch.no_grad():
    node_emb, _ = tgnn(graph_batch)
    node_tokens = pad_node_embeddings(node_emb, graph_batch.batch, N_GRAPH_TOKENS, 256).to(dtype_t)
    combined_embeds, combined_mask = adapter(node_tokens, prompt_embeds, attn_mask)

print(f"  node_tokens:      {node_tokens.shape}")      # [1, 32, 256]
print(f"  combined_embeds:  {combined_embeds.shape}")  # [1, 32+seq, 4096]
print(f"  combined_mask:    {combined_mask.shape}")    # [1, 32+seq]
check(combined_embeds.shape[1] == N_GRAPH_TOKENS + prompt_embeds.shape[1],
      f"combined seq = {N_GRAPH_TOKENS} graph + {prompt_embeds.shape[1]} text")

# ── 7. Transformer forward ────────────────────────────────────────────────
print("\n=== 7. Transformer forward ===")
with torch.no_grad():
    v_pred = pipe.transformer(
        hidden_states          = noisy,
        encoder_hidden_states  = combined_embeds,
        pooled_projections     = pooled_embeds,
        timestep               = timesteps,
        encoder_attention_mask = combined_mask,
        guidance               = guidance,   # required when guidance_embeds=True
        return_dict            = False,
    )[0]
print(f"  v_pred: {v_pred.shape}")
check(v_pred.shape == latents.shape, "v_pred shape matches latents")

# ── 8. Loss ───────────────────────────────────────────────────────────────
print("\n=== 8. Loss ===")
loss = F.mse_loss(v_pred.float(), v_target.float())
print(f"  MSE loss: {loss.item():.4f}")
check(torch.isfinite(loss), "loss finite")

print("\n=== HunyuanVideo Dry-run complete — all shapes verified ===\n")
print("Ready for training. Submit run_train_hunyuan.sh")
