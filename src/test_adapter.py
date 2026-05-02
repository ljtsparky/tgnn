"""
test_adapter.py
===============
Smoke-test for GraphCondAdapter + TGNNEncoder integration.

Checks:
  1. Adapter output shape [B, N, 4096]
  2. Zero-init: graph_tokens are all zeros before any training step
  3. Concatenation with mock text embeddings works
  4. Gradient flows through adapter + tgnn only (not into frozen transformer proxy)
  5. Parameter counts for both modules
"""

import sys
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from tgnn_model import TGNNEncoder
from adapter import GraphCondAdapter, build_graph_cond_modules
import os

GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
D_MODEL   = 256
N_TOKENS  = 4
D_TEXT    = 4096
TEXT_SEQ  = 226   # CogVideoX-2b default text sequence length

def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)

# ---------------------------------------------------------------------------
print("\n=== Loading 4 graphs ===")
graphs = []
for pt in sorted(os.listdir(GRAPH_DIR))[:4]:
    graphs.append(torch.load(os.path.join(GRAPH_DIR, pt), weights_only=False))
batch = Batch.from_data_list(graphs)
B = 4
print(f"  Batch size : {B}")

# ---------------------------------------------------------------------------
print("\n=== 1. Build modules ===")
tgnn, adapter = build_graph_cond_modules(
    d_model=D_MODEL, n_layers=3, n_heads=8, n_tokens=N_TOKENS
)
tgnn.eval(); adapter.eval()

n_tgnn    = sum(p.numel() for p in tgnn.parameters())
n_adapter = sum(p.numel() for p in adapter.parameters())
print(f"  TGNNEncoder params    : {n_tgnn:,}")
print(f"  GraphCondAdapter params: {n_adapter:,}")
print(f"  Total trainable       : {n_tgnn + n_adapter:,}")
check(n_adapter > 0, "adapter has parameters")

# ---------------------------------------------------------------------------
print("\n=== 2. Zero-init check ===")
with torch.no_grad():
    _, graph_emb = tgnn(batch)           # [4, 256]
    graph_tokens = adapter(graph_emb)    # [4, N_TOKENS, D_TEXT]
print(f"  graph_emb shape   : {graph_emb.shape}")
print(f"  graph_tokens shape: {graph_tokens.shape}")
check(graph_tokens.shape == (B, N_TOKENS, D_TEXT), "graph_tokens shape correct")
check(graph_tokens.abs().max().item() == 0.0, "zero-init: graph_tokens all zero at init")

# ---------------------------------------------------------------------------
print("\n=== 3. Concatenation with text embeddings ===")
text_emb = torch.randn(B, TEXT_SEQ, D_TEXT)
cond = torch.cat([graph_tokens, text_emb], dim=1)
print(f"  text_emb shape : {text_emb.shape}")
print(f"  cond shape     : {cond.shape}   (expected [{B}, {TEXT_SEQ + N_TOKENS}, {D_TEXT}])")
check(cond.shape == (B, TEXT_SEQ + N_TOKENS, D_TEXT), "cond shape correct after prepend")

# ---------------------------------------------------------------------------
print("\n=== 4. Gradient flow (adapter + tgnn only) ===")
tgnn.train(); adapter.train()

# Proxy for frozen transformer: just a linear on cond
frozen_proxy = nn.Linear(D_TEXT, 1)
for p in frozen_proxy.parameters():
    p.requires_grad_(False)

_, graph_emb = tgnn(batch)
graph_tokens = adapter(graph_emb)
text_emb     = torch.randn(B, TEXT_SEQ, D_TEXT)
cond         = torch.cat([graph_tokens, text_emb], dim=1)
loss         = frozen_proxy(cond).sum()   # simulates transformer output
loss.backward()

tgnn_grads    = [p.grad for p in tgnn.parameters()    if p.requires_grad]
adapter_grads = [p.grad for p in adapter.parameters() if p.requires_grad]
proxy_grads   = [p.grad for p in frozen_proxy.parameters()]

tgnn_ok    = all(g is not None and torch.isfinite(g).all() for g in tgnn_grads)
adapter_ok = all(g is not None and torch.isfinite(g).all() for g in adapter_grads)
proxy_ok   = all(g is None for g in proxy_grads)  # frozen → no grad

check(tgnn_ok,    "TGNNEncoder gradients finite")
check(adapter_ok, "GraphCondAdapter gradients finite")
check(proxy_ok,   "frozen transformer receives no gradients")

# ---------------------------------------------------------------------------
print("\n=== All checks passed ===\n")
print(f"Summary:")
print(f"  graph_emb  : [B={B}, d={D_MODEL}]")
print(f"  graph_tokens : [B={B}, N={N_TOKENS}, d_text={D_TEXT}]")
print(f"  cond (prepended) : [B={B}, seq={TEXT_SEQ + N_TOKENS}, d_text={D_TEXT}]")
