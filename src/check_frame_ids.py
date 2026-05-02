"""
check_frame_ids.py
Inspect frame_ids from a sample graph to confirm naming format.
"""
import torch

d = torch.load('/projects/bgnv/leatherman/tgnn/dataset/ag/graphs/001YG.pt', weights_only=False)
print('video_id  :', d.video_id)
print('num_frames:', d.num_frames)
print('frame_ids :', d.frame_ids.tolist())
