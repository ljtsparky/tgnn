"""下载 Wan2.1-T2V-1.3B 到 HF cache"""
import os
os.environ["HF_HOME"] = "/projects/bgnv/leatherman/hf_cache"
from huggingface_hub import snapshot_download
print("Downloading HunyuanVideo ...")
path = snapshot_download("hunyuanvideo-community/HunyuanVideo")
print(f"Downloaded to: {path}")
