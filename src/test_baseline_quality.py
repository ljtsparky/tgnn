"""
test_baseline_quality.py
========================
测试 CogVideoX-2b 原始推理质量（无任何 adapter）：
  A) 256×256（之前训练用的分辨率，怀疑是噪声来源）
  B) 480×720（模型设计的原始分辨率）

相同 prompt、相同 seed，对比找出视频质量差的根因。
"""

import os, torch
os.environ["HF_HOME"] = "/projects/bgnv/leatherman/hf_cache"

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

OUT = "/projects/bgnv/leatherman/tgnn/outputs"
os.makedirs(OUT, exist_ok=True)

device = torch.device("cuda")
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
pipe = pipe.to(device)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

PROMPT = "a person in an indoor scene"
SEED   = 42

for label, h, w in [("256x256", 256, 256), ("480x720", 480, 720)]:
    print(f"\nGenerating {label} ...")
    out = pipe(
        prompt=PROMPT, negative_prompt="",
        height=h, width=w, num_frames=49,
        num_inference_steps=50, guidance_scale=6.0,
        generator=torch.Generator(device).manual_seed(SEED),
    )
    path = f"{OUT}/baseline_{label}.mp4"
    export_to_video(out.frames[0], path, fps=8)
    print(f"  Saved → {path}")

print("\nDone. Compare baseline_256x256.mp4 vs baseline_480x720.mp4")
print("If 256x256 is noise but 480x720 is clear: resolution is the problem.")
print("If both are garbage: something else is wrong with inference setup.")
