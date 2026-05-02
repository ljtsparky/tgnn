import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import os

output_dir = "/projects/bgnv/leatherman/cogvideox_output"
os.makedirs(output_dir, exist_ok=True)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-2b",
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

output = pipe(
    prompt="a cat walking on a sunny beach, high quality, smooth motion",
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator("cuda").manual_seed(42),
)

video_path = os.path.join(output_dir, "output.mp4")
export_to_video(output.frames[0], video_path, fps=8)
print(f"Video saved to {video_path}")