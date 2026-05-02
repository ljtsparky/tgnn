"""
adapter_wan_v2.py
=================
WanGraphAdapterV2 — correct cross-attention injection for Wan2.1.

Why v1 failed
-------------
v1 prepended graph tokens [B, 32, 4096] to T5 tokens [B, 512, 4096] in the
T5 embedding space, then passed them as prompt_embeds. Inside WanTransformer3DModel,
condition_embedder.text_embedder applied PixArtAlphaTextProjection (4096→1536) to
ALL tokens. Our graph tokens don't live in T5's semantic space, so the projection
produced garbage — causing color-block generation.

Correct injection point
-----------------------
WanTransformer3DModel.forward():
  ...
  temb, timestep_proj, encoder_hs, _ = self.condition_embedder(
      timestep, encoder_hidden_states   # encoder_hs: T5 → 1536-dim
  )
  for block in self.blocks:
      hidden_states = block(hidden_states, encoder_hs, ...)  # cross-attn K/V

We hook on condition_embedder's OUTPUT, where encoder_hs is already 1536-dim.
Then we project graph tokens 256→1536 and PREPEND to encoder_hs.
All 30 blocks' cross-attention then sees [graph_tokens | text_tokens] in the
correct inner-dim space.

This is exactly how Wan2.1-I2V injects image tokens (via encoder_hidden_states_image),
just adapted for our graph structure.

Wan analogy
-----------
  Wan I2V:  image_embedder(image_features) → 1536-dim → concat to encoder_hs
  Ours:     graph_proj(node_tokens)        → 1536-dim → prepend to encoder_hs

Parameters (1.3B model, inner_dim=1536)
-----------------------------------------
  TGNNEncoder          :   796K
  WanGraphAdapterV2:
    input_norm(256)    :     512
    graph_proj(256→1536→1536): 393K + 2,360K = 2,754K
  Total trainable      : ~3.55M   (lightest so far)

Mixed training strategy
-----------------------
During training, some batches have real graphs (Action Genome) and some have
null graphs (large-scale backbone data). graph_tokens=None → hook returns
unmodified encoder_hs → model generates text-only, preserving Wan's quality.

Graph dropout: randomly set graph_tokens=None with prob p_drop during training.
Enables classifier-free guidance at inference:
    cond    = adapter(graph_tokens, ...)  → graph-conditioned
    uncond  = adapter(None, ...)          → text-only (no graph)
    output  = uncond + guidance_scale * (cond - uncond)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class WanGraphAdapterV2(nn.Module):
    """
    Hook-based graph conditioning for Wan2.1.
    Injects graph tokens into condition_embedder output (1536-dim space).

    Parameters
    ----------
    transformer     : WanTransformer3DModel (frozen)
    d_graph         : TGNNEncoder node_emb dimension (default 256)
    n_graph_tokens  : padded node tokens per graph (default 32)
    d_inner         : Wan transformer inner dim (1536 for 1.3B, 5120 for 14B)
    """

    def __init__(
        self,
        transformer,
        d_graph:        int = 256,
        n_graph_tokens: int = 32,
        d_inner:        int = 1536,
    ) -> None:
        super().__init__()
        self.n_graph_tokens = n_graph_tokens
        self.d_inner        = d_inner
        self._graph_tokens: Tensor | None = None

        self.input_norm = nn.LayerNorm(d_graph)

        # Project graph node tokens [B, N, d_graph] → [B, N, d_inner]
        # Two-layer MLP: d_graph → d_inner (hidden) → d_inner (zero-init output)
        self.graph_proj = nn.Sequential(
            nn.Linear(d_graph, d_inner),
            nn.SiLU(),
            nn.Linear(d_inner, d_inner),
        )
        # Zero-init output: at init, graph projection = 0 → no graph influence
        nn.init.zeros_(self.graph_proj[-1].weight)
        nn.init.zeros_(self.graph_proj[-1].bias)

        # Register hook on condition_embedder (fires after text_embedder projection)
        self._hook = transformer.condition_embedder.register_forward_hook(
            self._inject_hook
        )

    def _inject_hook(self, module, inputs, output):
        """
        Fires after condition_embedder.
        output = (temb, timestep_proj, encoder_hs, encoder_hs_img)
        encoder_hs is now in 1536-dim (already projected from T5's 4096-dim).
        """
        if self._graph_tokens is None:
            return output   # null graph → pass through unmodified (text-only)

        temb, timestep_proj, encoder_hs, encoder_hs_img = output
        B_hs = encoder_hs.size(0)

        gt = self._graph_tokens
        # CFG: pipeline doubles batch (cond + uncond); tile graph tokens
        if gt.size(0) != B_hs:
            gt = gt.repeat(B_hs // gt.size(0), 1, 1)

        # Cast graph tokens to match encoder_hs dtype, then project
        x = self.input_norm(gt.to(encoder_hs.dtype))
        graph_cond = self.graph_proj(x)             # [B, N, d_inner]

        # Prepend graph tokens to text tokens (both in d_inner space)
        encoder_hs = torch.cat([graph_cond, encoder_hs], dim=1)   # [B, N+512, d_inner]

        return temb, timestep_proj, encoder_hs, encoder_hs_img

    def set_graph_tokens(self, graph_tokens: Tensor) -> None:
        """Set graph tokens before transformer forward. [B, N, d_graph]"""
        self._graph_tokens = graph_tokens

    def clear_graph_tokens(self) -> None:
        self._graph_tokens = None

    def remove_hook(self) -> None:
        self._hook.remove()

    def forward(self, *args, **kwargs):
        raise RuntimeError("Don't call adapter.forward(). Use set_graph_tokens() then run transformer.")
