"""
test_train_dryrun.py
====================
Dry-run test: loads all pipeline components + runs ONE forward pass
to verify shapes before committing to a full training run.

Uses cross-attention injection (GraphCrossAttnAdapter) — not FiLM.
"""

import sys, os
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from diffusers import CogVideoXPipeline
from diffusers import DDPMScheduler

sys.path.insert(0, os.path.dirname(__file__))
from tgnn_model import TGNNEncoder
from adapter    import GraphCrossAttnAdapter, pad_node_embeddings
from train      import AGTrainDataset, collate_fn, encode_prompt, encode_frames, TrainConfig

FRAMES_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/frames"
HF_CACHE   = "/projects/bgnv/leatherman/hf_cache"

N_GRAPH_TOKENS = 32

def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)

os.environ["HF_HOME"] = HF_CACHE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float16 if device.type == "cuda" else torch.float32
print(f"\nDevice: {device}  dtype: {dtype}")

# ---------------------------------------------------------------------------
print("\n=== 1. Load pipeline (frozen) ===")
pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=dtype)
vae          = pipe.vae.to(device).eval()
text_encoder = pipe.text_encoder.to(device).eval()
tokenizer    = pipe.tokenizer
transformer  = pipe.transformer.to(device).eval()
for m in [vae, text_encoder, transformer]:
    m.requires_grad_(False)
vae.enable_slicing()
vae.enable_tiling()
print("  Pipeline loaded.")
print(f"  n_transformer_blocks : {len(transformer.transformer_blocks)}")
print(f"  VAE scaling_factor   : {vae.config.scaling_factor}")

# ---------------------------------------------------------------------------
print("\n=== 2. Load graph + frames ===")
cfg = TrainConfig()
cfg.n_frames   = 49
cfg.frame_h    = 256
cfg.frame_w    = 256
cfg.frames_dir = FRAMES_DIR
ds = AGTrainDataset(cfg)

from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=True)
graph_batch, frames, prompts = next(iter(loader))
graph_batch = graph_batch.to(device)
frames      = frames.to(device)
B = frames.size(0)
print(f"  Batch size     : {B}")
print(f"  frames shape   : {frames.shape}")
print(f"  Prompts        : {prompts}")
check(frames.shape[0] == 1, "B=1")
check(frames.shape[1] == 49, "T=49 frames")

# ---------------------------------------------------------------------------
print("\n=== 3. VAE encode ===")
latents = encode_frames(frames, vae, device, dtype)
print(f"  latents shape (B,T,C,H,W) : {latents.shape}")
check(latents.ndim == 5, "latents is 5-D")
T_lat = latents.shape[1]
print(f"  T_lat = {T_lat}")

# ---------------------------------------------------------------------------
print("\n=== 4. Noise addition ===")
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise     = torch.randn_like(latents)
timesteps = torch.randint(0, 1000, (B,), device=device).long()
noisy     = noise_scheduler.add_noise(latents, noise, timesteps)
check(noisy.shape == latents.shape, "noisy latents shape matches latents")

# ---------------------------------------------------------------------------
print("\n=== 5. Text encode ===")
text_emb = encode_prompt(prompts, tokenizer, text_encoder, device=device, dtype=dtype)
print(f"  text_emb shape : {text_emb.shape}")
check(text_emb.shape == (B, 226, 4096), f"text_emb [{B}, 226, 4096]")

# ---------------------------------------------------------------------------
print("\n=== 6. TGNNEncoder + GraphCrossAttnAdapter ===")
tgnn    = TGNNEncoder(d_model=256).to(device).eval()
adapter = GraphCrossAttnAdapter(
    transformer    = transformer,
    d_graph        = 256,
    n_graph_tokens = N_GRAPH_TOKENS,
    d_cross        = 256,
    n_heads        = 8,
).to(device).eval()

n_params = sum(p.numel() for p in list(tgnn.parameters()) + list(adapter.parameters()))
print(f"  Total trainable params: {n_params:,}")

with torch.no_grad():
    node_emb, graph_emb = tgnn(graph_batch)
    graph_tokens = pad_node_embeddings(
        node_emb, graph_batch.batch, N_GRAPH_TOKENS, 256
    ).to(dtype)
    adapter.set_graph_tokens(graph_tokens)

print(f"  node_emb     : {node_emb.shape}")
print(f"  graph_emb    : {graph_emb.shape}")
print(f"  graph_tokens : {graph_tokens.shape}")
check(graph_emb.shape == (B, 256),              "graph_emb [B, 256]")
check(graph_tokens.shape == (B, N_GRAPH_TOKENS, 256), f"graph_tokens [{B}, {N_GRAPH_TOKENS}, 256]")

# ---------------------------------------------------------------------------
print("\n=== 7. Transformer forward (cross-attn hooks active) ===")
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=dtype):
        pred = transformer(
            hidden_states         = noisy,
            encoder_hidden_states = text_emb,   # original 226 tokens, no FiLM
            timestep              = timesteps,
            return_dict           = False,
        )[0]
adapter.clear_graph_tokens()
print(f"  pred shape : {pred.shape}")
check(pred.shape == latents.shape, "pred shape matches latents")

# ---------------------------------------------------------------------------
print("\n=== 8. Loss ===")
loss = F.mse_loss(pred.float(), noise.float())
print(f"  MSE loss : {loss.item():.4f}")
check(torch.isfinite(loss), "loss is finite")

# ---------------------------------------------------------------------------
print("\n=== Dry-run complete — all shapes verified ===\n")
print("Ready for full training. Submit run_train.sh to start.")
