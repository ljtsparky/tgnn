"""
adapter_hunyuan.py
==================
HunyuanGraphAdapter — graph conditioning adapter for HunyuanVideo.

Architecture overview
---------------------
HunyuanVideo uses a dual text encoder:
  text_encoder  (LLaMA-3-8B): sequence embeddings  [B, seq, 4096]
  text_encoder_2 (CLIP):       pooled projection    [B, 768]

The transformer (20 double-stream + 40 single-stream blocks) takes:
  encoder_hidden_states: LLaMA sequence tokens   [B, seq, 4096]
  pooled_projections:    CLIP pooled             [B, 768]

Injection strategy (no hooks needed!)
--------------------------------------
Unlike Wan (which needed a hook on condition_embedder), HunyuanVideoPipeline
accepts prompt_embeds directly in __call__. We simply prepend graph tokens
to the LLaMA sequence before passing to the pipeline:

  graph_cond [B, N, 4096]  ←  adapter.graph_proj(node_tokens)
  text_emb   [B, seq, 4096] ←  LLaMA text encoder output
  ──────────────────────────────────────────────────────────
  combined   [B, N+seq, 4096] → passed as prompt_embeds
  combined_mask [B, N+seq]    → extend attention_mask with 1s for graph tokens

The CLIP pooled projection is NOT modified (it's used for timestep
conditioning, not content conditioning).

Why this works
--------------
HunyuanVideo's double-stream blocks process encoder_hidden_states and
video tokens with separate but interacting attention streams. By prepending
graph tokens, EVERY double-stream block attends to graph information.
In single-stream blocks, encoder_hidden_states are concatenated with
video tokens, so graph tokens participate in the merged attention too.

Zero-init: graph_proj final layer is zero-initialized → at init, graph_cond=0
→ combined_emb starts as purely text conditioning → stable training start.

Parameters
----------
  input_norm  : LayerNorm(256)                    → 512
  graph_proj  : Linear(256,1024) + SiLU           → 263K
              + Linear(1024,4096), zero-init       → 4.2M
  Total WanGraphAdapter: ~4.5M
  + TGNNEncoder:         ~796K
  = Total trainable:     ~5.3M
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

HUNYUAN_TEXT_DIM = 4096   # LLaMA-3-8B hidden_size = text_embed_dim in transformer config


class HunyuanGraphAdapter(nn.Module):
    """
    Projects graph node tokens into LLaMA embedding space and prepends
    to the prompt_embeds sequence passed to HunyuanVideoPipeline.

    Parameters
    ----------
    d_graph       : TGNNEncoder node_emb dim (default 256)
    d_text        : LLaMA hidden_size = transformer text_embed_dim (default 4096)
    n_graph_tokens: padded node tokens per graph (default 32)
    """

    def __init__(
        self,
        d_graph:        int = 256,
        d_text:         int = HUNYUAN_TEXT_DIM,
        n_graph_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.n_graph_tokens = n_graph_tokens
        self.d_text         = d_text

        self.input_norm = nn.LayerNorm(d_graph)

        # MLP: 256 → 1024 → 4096 (same as Wan adapter, d_text=4096)
        hidden = d_graph * 4  # 1024
        self.graph_proj = nn.Sequential(
            nn.Linear(d_graph, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_text),
        )

        # Zero-init last linear: at start, graph_cond = 0 → text-only baseline
        nn.init.zeros_(self.graph_proj[-1].weight)
        nn.init.zeros_(self.graph_proj[-1].bias)

    def forward(
        self,
        node_tokens:         Tensor,              # [B, N, d_graph]  — padded node embs
        prompt_embeds:       Tensor,              # [B, seq, d_text] — LLaMA output
        prompt_attention_mask: Tensor | None = None,  # [B, seq] or None
    ) -> tuple[Tensor, Tensor]:
        """
        Prepend graph tokens to LLaMA sequence.

        Returns
        -------
        combined_embeds : [B, N + seq, d_text]  — graph tokens prepended to text
        combined_mask   : [B, N + seq]          — 1s for graph tokens + original mask
        """
        B = node_tokens.size(0)
        N = self.n_graph_tokens

        # Project: [B, N, 256] → [B, N, 4096]
        x           = self.input_norm(node_tokens)       # normalize first
        graph_cond  = self.graph_proj(x)                 # [B, N, d_text]

        # Prepend graph tokens to LLaMA sequence
        combined_embeds = torch.cat([graph_cond, prompt_embeds], dim=1)  # [B, N+seq, d_text]

        # Extend attention mask: graph tokens are all valid (mask=1)
        if prompt_attention_mask is not None:
            graph_mask      = torch.ones(B, N, dtype=prompt_attention_mask.dtype,
                                         device=prompt_attention_mask.device)
            combined_mask   = torch.cat([graph_mask, prompt_attention_mask], dim=1)  # [B, N+seq]
        else:
            seq = prompt_embeds.size(1)
            combined_mask = torch.ones(B, N + seq, device=prompt_embeds.device,
                                       dtype=torch.long)

        return combined_embeds, combined_mask
