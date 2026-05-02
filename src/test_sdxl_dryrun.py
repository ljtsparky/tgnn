"""
test_sdxl_dryrun.py
====================
Verify shapes through full forward pass — no backward.
"""
import sys, os, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HOME"] = "/projects/bgnv/leatherman/hf_cache"

from torch.utils.data import DataLoader
from diffusers import StableDiffusionXLPipeline, DDPMScheduler

from tgnn_model    import TGNNEncoder
from adapter_sdxl  import SDXLGraphAdapter, pad_node_embeddings_temporal
from dataset_image import ImageDataConfig, MixedImageDataset, AGImageDataset, collate_fn_image
from train_sdxl    import encode_image_sdxl, encode_prompt_sdxl, get_add_time_ids

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
device = torch.device("cuda")
dtype  = torch.bfloat16

def check(c, msg):
    print(f"  [{'PASS' if c else 'FAIL'}] {msg}")
    if not c: sys.exit(1)

print("\n=== 1. Load SDXL ===")
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID, torch_dtype=dtype, local_files_only=True,
).to(device)
pipe.vae.to(torch.float32)                                       # fp16 VAE = NaN
cross_dim = pipe.unet.config.cross_attention_dim
print(f"  cross_dim={cross_dim}")
check(cross_dim == 2048, "cross_dim=2048")

print("\n=== 2. Build dataset (val for speed) ===")
cfg = ImageDataConfig(image_h=1024, image_w=1024, graph_prob=1.0, split="val")
ds  = AGImageDataset(cfg)
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                    collate_fn=collate_fn_image, drop_last=True)
batch = next(iter(loader))
graph_batch, images, target_idxs, prompts, has_graph = batch
print(f"  images={images.shape}  target_idxs={target_idxs.tolist()}  prompts={prompts}")
check(images.shape == (2,3,1024,1024), "image batch [2,3,1024,1024]")

print("\n=== 3. VAE encode ===")
images = images.to(device)
latents = encode_image_sdxl(images, pipe.vae, dtype)
print(f"  latents={latents.shape}")
check(latents.shape[1] == 4 and latents.shape[-1] == 128, "latents [B,4,128,128]")

print("\n=== 4. Text encode ===")
prompt_embeds, pooled = encode_prompt_sdxl(pipe, prompts, device, dtype)
print(f"  prompt={prompt_embeds.shape}  pooled={pooled.shape}")
check(prompt_embeds.shape[-1] == 2048, "prompt_dim=2048")

print("\n=== 5. TGNN + adapter ===")
tgnn = TGNNEncoder(d_model=256).to(device).eval()
adapter = SDXLGraphAdapter(d_graph=256, cross_dim=cross_dim).to(device, dtype).eval()
adapter.install(pipe.unet)
n_p = sum(p.numel() for p in list(tgnn.parameters()) + list(adapter.parameters()))
print(f"  trainable params: {n_p:,}")

graph_batch = graph_batch.to(device)
target_idxs = target_idxs.to(device)
with torch.no_grad():
    node_emb, _, is_target = tgnn(graph_batch, target_idxs)
    node_tokens = pad_node_embeddings_temporal(
        node_emb, graph_batch.batch, graph_batch.node_frame, target_idxs,
        is_target, n_tokens=64, d_graph=256,
    ).to(dtype)
    graph_tokens = adapter.encode(node_tokens)
print(f"  node_emb={node_emb.shape}  node_tokens={node_tokens.shape}  graph_tokens={graph_tokens.shape}")
check(node_tokens.shape == (2,64,256), "node_tokens [2,64,256]")
check(graph_tokens.shape == (2,64,2048), "graph_tokens [2,64,2048]")

print("\n=== 6. UNet forward ===")
adapter.set_graph_tokens(graph_tokens)
scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler", local_files_only=True)
noise = torch.randn_like(latents)
ts = torch.randint(0, 1000, (2,), device=device, dtype=torch.long)
noisy = scheduler.add_noise(latents, noise, ts)
add_time_ids = get_add_time_ids(1024, 1024, dtype, 2, device)
with torch.no_grad():
    pred = pipe.unet(
        noisy, ts, encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
        return_dict=False,
    )[0]
adapter.clear_graph_tokens()
print(f"  noise_pred={pred.shape}")
check(pred.shape == latents.shape, "noise_pred matches latents")

print("\n=== 7. Loss ===")
loss = F.mse_loss(pred.float(), noise.float())
print(f"  loss={loss.item():.5f}")
check(torch.isfinite(loss), "loss finite")

print("\n=== Dry-run all PASS ===\n")
