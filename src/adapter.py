"""
adapter.py
==========
GraphCrossAttnAdapter — IP-Adapter style cross-attention injection for CogVideoX.

vs. FiLM (old approach)
------------------------
FiLM: graph_emb [B,256] → scale+shift → modulates text_emb globally.
      One vector carries all graph info. Breaks if adapter hasn't converged.

Cross-attention (this approach)
--------------------------------
TGNNEncoder → node_emb [N_nodes, 256]  (per-node embeddings, not collapsed)
            ↓ pad to fixed size
            graph_tokens [B, N_graph_tokens, 256]

In each CogVideoXBlock (via PyTorch forward hook):
  video_patches [B, T·H·W, 1920]  ← Query
      × graph_tokens K, V         ← Key, Value (projected from node_emb)
  → cross-attn output             ← zero-init → starts as identity
  → video_patches += output

This means every video patch token directly attends to every graph node token,
capturing fine-grained spatial and relational structure. CLIP-score advantage:
the video layout can align with which objects are where, not just the summary.

Design decisions
----------------
1. n_graph_tokens=32: covers Action Genome mean (~18 nodes) + large graphs.
   Nodes are sorted by frame_id, so the first 32 capture the whole temporal range
   for most videos.

2. d_cross=256, n_heads=8, head_dim=32:  compact attention (not full 1920-dim),
   keeps per-block param count ~984K. 42 blocks → ~41M total. Feasible on A100.

3. Shared K, V projectors (not per-block): reduces params, stabilises training.
   Per-block: Q projection + LayerNorm + out projection (zero-init).

4. Zero-init to_out: at init cross-attn output = 0 → text-only baseline.
   Graph conditioning learned incrementally.

5. Hook-based injection: no modification to CogVideoX source needed.
   Hook intercepts each CogVideoXBlock output: (video_hs, text_hs).
   Only video_hs is modified; text_hs unchanged.

6. CFG-aware: during pipeline inference with guidance_scale>1, the pipeline
   concatenates cond+uncond into B=2 batch. Hook detects B mismatch and tiles
   graph_tokens automatically (both passes get same graph, only text differs).

Parameters (CogVideoX-2b, 42 blocks)
--------------------------------------
  TGNNEncoder          :   796K
  Shared KV projectors :   131K   (2 × 256×256)
  42 × cross-attn block: 41.3M   (42 × [LN + to_q + to_out])
  ────────────────────────────────
  Total trainable      : ~42.2M
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

COGVIDEOX_INNER_DIM = 1920   # 30 heads × 64 head_dim (CogVideoX-2b)


# ---------------------------------------------------------------------------
# Utility: pad per-graph node embeddings to fixed shape
# ---------------------------------------------------------------------------

def pad_node_embeddings(
    node_emb: Tensor,      # [N_total, d_graph]  — PyG batched nodes
    batch_idx: Tensor,     # [N_total]            — graph index per node
    n_tokens: int,
    d_graph: int,
) -> Tensor:
    """Return [B, n_tokens, d_graph] — padded (zeros) or truncated per graph."""
    B = int(batch_idx.max().item()) + 1
    out = torch.zeros(B, n_tokens, d_graph, device=node_emb.device, dtype=node_emb.dtype)
    for i in range(B):
        mask = (batch_idx == i)
        nodes = node_emb[mask]          # [N_i, d_graph]
        n = min(nodes.size(0), n_tokens)
        out[i, :n] = nodes[:n]
    return out


# ---------------------------------------------------------------------------
# Per-block cross-attention layer
# ---------------------------------------------------------------------------

class GraphCrossAttnLayer(nn.Module):
    """
    Single cross-attention: video patches (Q) attend to graph nodes (K, V).
    K and V projectors are shared across all blocks (passed in at forward time).
    """

    def __init__(self, d_hidden: int = COGVIDEOX_INNER_DIM, d_cross: int = 256, n_heads: int = 8) -> None:
        super().__init__()
        assert d_cross % n_heads == 0
        self.n_heads   = n_heads
        self.head_dim  = d_cross // n_heads
        self.scale     = self.head_dim ** -0.5
        self.d_cross   = d_cross

        self.norm  = nn.LayerNorm(d_hidden)
        self.to_q  = nn.Linear(d_hidden, d_cross, bias=False)
        self.to_out = nn.Linear(d_cross, d_hidden, bias=False)

        # Zero-init: cross-attn output = 0 at start → text-only baseline
        nn.init.zeros_(self.to_out.weight)

    def forward(self, video_hs: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        video_hs : [B, L, d_hidden]   — video patch tokens from CogVideoXBlock
        k        : [B, N, d_cross]    — graph nodes projected to key space
        v        : [B, N, d_cross]    — graph nodes projected to value space
        returns  : [B, L, d_hidden]   — modulated video_hs
        """
        B, L, _ = video_hs.shape
        N = k.size(1)

        residual = video_hs
        x = self.norm(video_hs)

        Q = self.to_q(x)                                          # [B, L, d_cross]
        Q = Q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)   # [B, h, L, hd]
        K = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)   # [B, h, N, hd]
        V = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)   # [B, h, N, hd]

        attn = F.scaled_dot_product_attention(Q, K, V)            # [B, h, L, hd]
        attn = attn.transpose(1, 2).reshape(B, L, self.d_cross)   # [B, L, d_cross]

        return residual + self.to_out(attn)


# ---------------------------------------------------------------------------
# Full adapter: manages all layers + hooks + shared projectors
# ---------------------------------------------------------------------------

class GraphCrossAttnAdapter(nn.Module):
    """
    Registers one GraphCrossAttnLayer per CogVideoXBlock via forward hooks.
    Call set_graph_tokens(graph_tokens) before each transformer forward pass.

    Parameters
    ----------
    transformer     : CogVideoXTransformer3DModel (frozen)
    d_graph         : TGNNEncoder node embedding dim (default 256)
    n_graph_tokens  : number of padded node tokens per graph (default 32)
    d_cross         : internal cross-attention dim (default 256)
    n_heads         : number of attention heads in cross-attention (default 8)
    """

    def __init__(
        self,
        transformer,
        d_graph:        int = 256,
        n_graph_tokens: int = 32,
        d_cross:        int = 256,
        n_heads:        int = 8,
        d_hidden:       int = COGVIDEOX_INNER_DIM,
    ) -> None:
        super().__init__()
        self.n_graph_tokens = n_graph_tokens
        self.d_graph        = d_graph
        self.d_cross        = d_cross

        n_blocks = len(transformer.transformer_blocks)

        # Shared K, V projectors — one set for all blocks
        self.to_k = nn.Linear(d_graph, d_cross, bias=False)
        self.to_v = nn.Linear(d_graph, d_cross, bias=False)

        # Per-block cross-attention layers
        self.layers = nn.ModuleList([
            GraphCrossAttnLayer(d_hidden=d_hidden, d_cross=d_cross, n_heads=n_heads)
            for _ in range(n_blocks)
        ])

        # Runtime state: set before each forward
        self._graph_tokens: Tensor | None = None

        # Register hooks (stored so we can remove them if needed)
        self._hooks = []
        for i, block in enumerate(transformer.transformer_blocks):
            layer = self.layers[i]
            h = block.register_forward_hook(self._make_hook(layer))
            self._hooks.append(h)

    def _make_hook(self, layer: GraphCrossAttnLayer):
        def hook(module, inputs, output):
            if self._graph_tokens is None:
                return output
            video_hs, text_hs = output
            B_hs = video_hs.size(0)

            # CFG batches cond+uncond together (B=2 when guidance_scale>1)
            gt = self._graph_tokens
            if gt.size(0) != B_hs:
                gt = gt.repeat(B_hs // gt.size(0), 1, 1)

            # Project graph tokens to K, V (cast to match video_hs dtype)
            k = self.to_k(gt.to(video_hs.dtype))   # [B, N, d_cross]
            v = self.to_v(gt.to(video_hs.dtype))   # [B, N, d_cross]

            video_hs = layer(video_hs, k, v)
            return video_hs, text_hs
        return hook

    def set_graph_tokens(self, graph_tokens: Tensor) -> None:
        """Call before each transformer forward. graph_tokens: [B, N, d_graph]."""
        self._graph_tokens = graph_tokens

    def clear_graph_tokens(self) -> None:
        self._graph_tokens = None

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def forward(self, *args, **kwargs):
        raise RuntimeError("Call set_graph_tokens() then run the transformer directly.")
