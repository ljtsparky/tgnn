"""
test_wan_dryrun.py
==================
Dry-run: verify all shapes for Wan2.1 + TGNNEncoder + WanGraphAdapter
before committing to full training. No backward pass.
"""

import sys, os
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from diffusers import WanPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model     import TGNNEncoder
from adapter_wan_v2 import WanGraphAdapterV2
from adapter        import pad_node_embeddings
from train_wan      import AGTrainDatasetWan, collate_fn, encode_frames_wan, encode_prompt_wan, WanTrainConfig

FRAMES_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
HF_CACHE   = "/projects/bgnv/leatherman/hf_cache"
N_GRAPH_TOKENS = 32

def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        sys.exit(1)

os.environ["HF_HOME"] = HF_CACHE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
print(f"\nDevice: {device}  dtype: {dtype}")

# ---------------------------------------------------------------------------
print("\n=== 1. Load Wan2.1 pipeline ===")
pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=dtype, local_files_only=True)
vae          = pipe.vae.to(device).eval()
text_encoder = pipe.text_encoder.to(device).eval()
tokenizer    = pipe.tokenizer
transformer  = pipe.transformer.to(device).eval()
for m in [vae, text_encoder, transformer]:
    m.requires_grad_(False)
vae.enable_slicing(); vae.enable_tiling()
inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
print(f"  Transformer blocks : {len(transformer.blocks)}")
print(f"  Transformer dim    : {inner_dim}  ({transformer.config.num_attention_heads} heads × {transformer.config.attention_head_dim})")
print(f"  text_dim (T5 out)  : {transformer.config.text_dim}")
d_text = transformer.config.text_dim  # 4096 for UMT5-XXL
print(f"  T5 output dim      : {d_text}")
print(f"  VAE temporal scale : {vae.config.scale_factor_temporal}")
print(f"  VAE spatial scale  : {vae.config.scale_factor_spatial}")

# ---------------------------------------------------------------------------
print("\n=== 2. Load graph + frames (480×832, 33 frames) ===")
cfg = WanTrainConfig()
cfg.n_frames   = 17
cfg.frame_h    = 480
cfg.frame_w    = 832
cfg.frames_dir = FRAMES_DIR
ds = AGTrainDatasetWan(cfg)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=collate_fn, drop_last=True)
graph_batch, frames, prompts = next(iter(loader))
graph_batch = graph_batch.to(device)
frames      = frames.to(device)
B = frames.size(0)
print(f"  frames shape : {frames.shape}")
print(f"  Prompt       : {prompts}")
check(frames.shape == (1, 17, 3, 480, 832), "frames [1, 17, 3, 480, 832]")

# ---------------------------------------------------------------------------
print("\n=== 3. VAE encode ===")
latents = encode_frames_wan(frames, vae, device, dtype)
print(f"  latents shape : {latents.shape}  (B, C, T_lat, H_lat, W_lat)")
check(latents.ndim == 5, "latents is 5-D")
check(latents.shape[0] == B, "batch dim correct")

# ---------------------------------------------------------------------------
print("\n=== 4. Flow matching noise ===")
noise     = torch.randn_like(latents)
sigma     = torch.rand(B, device=device, dtype=latents.dtype)
s         = sigma.view(B, 1, 1, 1, 1)
noisy     = (1.0 - s) * latents + s * noise
v_target  = noise - latents
timesteps = (sigma * 1000).long()
print(f"  noisy shape   : {noisy.shape}")
print(f"  sigma range   : [{sigma.min():.3f}, {sigma.max():.3f}]")
check(noisy.shape == latents.shape, "noisy shape matches latents")

# ---------------------------------------------------------------------------
print("\n=== 5. Text encode ===")
text_emb = encode_prompt_wan(prompts, tokenizer, text_encoder, device=device, dtype=dtype)
print(f"  text_emb shape : {text_emb.shape}")
check(text_emb.ndim == 3, "text_emb is 3-D")
check(text_emb.shape[0] == B, "batch dim correct")

# ---------------------------------------------------------------------------
print("\n=== 6. TGNNEncoder + WanGraphAdapterV2 (hook on condition_embedder) ===")
tgnn    = TGNNEncoder(d_model=256).to(device).eval()
adapter = WanGraphAdapterV2(
    transformer=transformer, d_graph=256, n_graph_tokens=N_GRAPH_TOKENS, d_inner=d_text // (4096 // 1536)
).to(device=device, dtype=dtype).eval()

# Actually compute d_inner correctly
d_inner = transformer.config.num_attention_heads * transformer.config.attention_head_dim
adapter = WanGraphAdapterV2(
    transformer=transformer, d_graph=256, n_graph_tokens=N_GRAPH_TOKENS, d_inner=d_inner,
).to(device=device, dtype=dtype).eval()

n_params = sum(p.numel() for p in list(tgnn.parameters()) + list(adapter.parameters()))
print(f"  Total trainable params : {n_params:,}")
print(f"  d_inner (injection dim): {d_inner}")

with torch.no_grad():
    node_emb, graph_emb = tgnn(graph_batch)
    node_tokens = pad_node_embeddings(node_emb, graph_batch.batch, N_GRAPH_TOKENS, 256).to(dtype)
    adapter.set_graph_tokens(node_tokens)

print(f"  node_emb    : {node_emb.shape}")
print(f"  node_tokens : {node_tokens.shape}  (will be injected in 1536-dim space)")
check(node_tokens.shape == (B, N_GRAPH_TOKENS, 256), f"node_tokens [{B}, {N_GRAPH_TOKENS}, 256]")

# ---------------------------------------------------------------------------
print("\n=== 7. Transformer forward (v2: hook injects in 1536-dim space) ===")
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=dtype):
        v_pred = transformer(
            hidden_states         = noisy,
            encoder_hidden_states = text_emb,  # T5 tokens; hook handles graph injection
            timestep              = timesteps,
            return_dict           = False,
        )[0]
adapter.clear_graph_tokens()
print(f"  v_pred shape : {v_pred.shape}")
check(v_pred.shape == latents.shape, "v_pred shape matches latents")

# ---------------------------------------------------------------------------
print("\n=== 8. Flow matching loss ===")
loss = F.mse_loss(v_pred.float(), v_target.float())
print(f"  MSE loss : {loss.item():.4f}")
check(torch.isfinite(loss), "loss is finite")

# ---------------------------------------------------------------------------
print("\n=== Wan2.1 Dry-run complete — all shapes verified ===\n")
print("Ready for full training. Submit run_train_wan.sh to start.")
