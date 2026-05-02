"""
test_tgnn_model.py
==================
Smoke-test for TGNNEncoder.

Checks:
  1. Forward pass on a single real graph — output shapes correct
  2. Forward pass on a PyG Batch of 4 graphs — batch dimension correct
  3. graph_emb is differentiable (backward does not crash)
  4. Parameter count printed for reference
"""

import sys
import torch
from torch_geometric.data import Batch
from tgnn_model import TGNNEncoder

GRAPH_DIR = "/projects/bgnv/leatherman/tgnn/dataset/ag/graphs"
D_MODEL   = 256

def check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    if not cond:
        sys.exit(1)

# ---------------------------------------------------------------------------
print("\n=== Loading sample graphs ===")
graphs = []
import os, random
pts = sorted(os.listdir(GRAPH_DIR))[:20]
for pt in pts:
    g = torch.load(os.path.join(GRAPH_DIR, pt), weights_only=False)
    graphs.append(g)
    if len(graphs) == 4:
        break
print(f"  Loaded {len(graphs)} graphs")

# ---------------------------------------------------------------------------
print("\n=== 1. Model init + param count ===")
model = TGNNEncoder(d_model=D_MODEL, n_layers=3, n_heads=8, dropout=0.1)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {n_params:,}")
check(n_params > 500_000, "model has >500K parameters")

# ---------------------------------------------------------------------------
print("\n=== 2. Single graph forward ===")
g = graphs[0]
with torch.no_grad():
    node_emb, graph_emb = model(g)
print(f"  Input  nodes      : {g.node_class.shape[0]}")
print(f"  node_emb shape    : {node_emb.shape}")
print(f"  graph_emb shape   : {graph_emb.shape}")
check(node_emb.shape  == (g.node_class.shape[0], D_MODEL), "node_emb shape correct")
check(graph_emb.shape == (1, D_MODEL),                      "graph_emb shape [1, d]")
check(node_emb.dtype  == torch.float32,                     "node_emb is float32")
check(torch.isfinite(node_emb).all(),                       "node_emb has no NaN/Inf")
check(torch.isfinite(graph_emb).all(),                      "graph_emb has no NaN/Inf")

# ---------------------------------------------------------------------------
print("\n=== 3. Batched forward (B=4) ===")
batch = Batch.from_data_list(graphs)
with torch.no_grad():
    node_emb_b, graph_emb_b = model(batch)
total_nodes = sum(g.node_class.shape[0] for g in graphs)
print(f"  Batch total nodes : {total_nodes}")
print(f"  node_emb shape    : {node_emb_b.shape}")
print(f"  graph_emb shape   : {graph_emb_b.shape}")
check(node_emb_b.shape  == (total_nodes, D_MODEL), "batched node_emb shape correct")
check(graph_emb_b.shape == (4, D_MODEL),            "batched graph_emb shape [B=4, d]")

# ---------------------------------------------------------------------------
print("\n=== 4. Backward pass (gradient flow) ===")
model.train()
node_emb_b, graph_emb_b = model(batch)
loss = graph_emb_b.sum()
loss.backward()
grads_ok = all(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in model.parameters() if p.requires_grad
)
check(grads_ok, "all parameter gradients finite after backward")

# ---------------------------------------------------------------------------
print("\n=== All checks passed ===\n")
