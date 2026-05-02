"""
adapter_wan.py
==============
WanGraphAdapter — graph conditioning adapter for Wan2.1.

Why Wan is better than CogVideoX for our task
----------------------------------------------
CogVideoX uses Joint Attention (text + video tokens merged before attention).
This caused pos_embedding size mismatch when we tried to prepend graph tokens.

Wan2.1 uses standard Cross-Attention: each WanTransformerBlock has a separate
cross-attention where video patches (Q) attend to text tokens (K/V).
There are NO positional embeddings on K/V — they're unconstrained key-value pairs.

=> We can simply prepend graph tokens to encoder_hidden_states.
   No hooks. No architecture modification. Wan handles variable-length K/V natively.

Injection point
---------------
  T5/UMT5 encodes prompt → text_emb [B, seq_len, 4096]
  TGNNEncoder → node_emb [N_total, 256]
              → pad → node_tokens [B, N_graph, 256]
              → WanGraphAdapter → graph_cond [B, N_graph, 4096]
  cond = cat([graph_cond, text_emb])  # [B, N_graph+seq_len, 4096]
  → WanPipeline(encoder_hidden_states=cond)
    Each WanTransformerBlock: video cross-attends to ALL tokens in cond

Zero-init: graph projection MLP final layer zero-initialised →
at init graph_cond == 0 → model starts from text-only baseline.

Parameters
----------
  TGNNEncoder (shared):  796K
  WanGraphAdapter:
    input_norm:           512   (LayerNorm(256))
    proj Linear×2:      256→1024→4096  =  262K + 4M = ~4.5M
  Total trainable:       ~5.3M   (much leaner than CogVideoX cross-attn at 30M)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

WAN_TEXT_DIM = 4096   # UMT5-XXL output dimension (Wan2.1-T2V-1.3B and 14B)


class WanGraphAdapter(nn.Module):
    """
    Projects graph node tokens into T5 embedding space and prepends to text_emb.

    Parameters
    ----------
    d_graph       : TGNNEncoder node_emb dimension (default 256)
    d_text        : Wan T5 text encoder output dim (default 4096)
    n_graph_tokens: number of padded node tokens per graph (default 32)
    """

    def __init__(
        self,
        d_graph:        int = 256,
        d_text:         int = WAN_TEXT_DIM,
        n_graph_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.n_graph_tokens = n_graph_tokens
        self.d_text         = d_text

        self.input_norm = nn.LayerNorm(d_graph)

        hidden = d_graph * 4   # 1024
        self.proj = nn.Sequential(
            nn.Linear(d_graph, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_text),
        )

        # Zero-init last linear: graph_cond == 0 at start → text-only baseline
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, node_tokens: Tensor, text_emb: Tensor) -> Tensor:
        """
        Parameters
        ----------
        node_tokens : [B, N_graph, d_graph]  — padded node embeddings from TGNNEncoder
        text_emb    : [B, seq_len, d_text]   — UMT5 text encoder output

        Returns
        -------
        cond : [B, N_graph + seq_len, d_text]
            Graph tokens prepended to text tokens.
            Passed as encoder_hidden_states to WanPipeline.
        """
        x = self.input_norm(node_tokens)        # [B, N, d_graph]
        graph_cond = self.proj(x)               # [B, N, d_text]
        return torch.cat([graph_cond, text_emb], dim=1)   # [B, N+seq, d_text]
