"""
adapter_sdxl.py
================
IP-Adapter-style graph conditioning for SDXL UNet.

Architecture
------------
For each cross-attention layer in the SDXL UNet, we replace the default
AttnProcessor with `GraphAttnProcessor` which performs TWO attention paths:

  1. Standard text cross-attention (Q from UNet, K/V from prompt embeds)
  2. Parallel graph cross-attention   (Q from UNet, K/V from graph tokens)

Outputs are summed:  out = text_attn(Q,K_t,V_t) + s * graph_attn(Q,K_g,V_g)

`s` is a learnable scalar per processor (init=0 → starts as text-only baseline,
matching the IP-Adapter convention).

Components
----------
  GraphProjector       : shared MLP 256→1024→cross_dim (e.g. 2048 for SDXL)
                         → [B, N_g, cross_dim]   the "graph image" tokens
  GraphAttnProcessor   : per-layer K_graph / V_graph projectors
                         applied as drop-in replacement of cross-attn processor

Why this works for SDXL specifically
------------------------------------
SDXL has ~70 cross-attention layers across the UNet. Each layer learns its
own graph K/V projectors → ~ (70 * 2 * D^2) params total.  At the deepest
attention dim (1280), that's ~70 * 2 * 1280^2 ≈ 230M params — too much.

We share the graph K/V projectors per RESOLUTION (Down1/2, Mid, Up1/2/3),
splitting roughly into 5 groups, drastically cutting parameter count to
~ 5 * 2 * 1280^2 ≈ 16M, plus the GraphProjector ~5M = ~21M total adapter.

IP-Adapter literature (Ye et al. 2023) validates this pattern on SD/SDXL
for image-prompt conditioning; we substitute "image features" with
"TGNN-encoded graph tokens".
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from diffusers.models.attention_processor import Attention
except Exception:
    Attention = None     # only used as a type hint at call time


# ---------------------------------------------------------------------------
# 1. Shared graph token projector (TGNN node_emb 256 → SDXL cross-dim)
# ---------------------------------------------------------------------------

class GraphProjector(nn.Module):
    """
    Projects TGNN node embeddings into SDXL's cross-attention dim space.

    Input  : node_tokens   [B, N_g, d_graph=256]
    Output : graph_tokens  [B, N_g, d_cross=2048]   (SDXL concat dim)

    NOT zero-init: we rely solely on zero-init `s` in each GraphAttnProcessor
    to start as text-only baseline. Double-zero-init (here AND scale=0) creates
    a fixed point where gradients to both sets of weights are 0 → adapter
    frozen at zero forever. Only ONE zero-init is needed.
    """

    def __init__(self, d_graph: int = 256, d_cross: int = 2048,
                 hidden_mult: int = 4) -> None:
        super().__init__()
        hidden = d_graph * hidden_mult
        self.norm = nn.LayerNorm(d_graph)
        self.proj = nn.Sequential(
            nn.Linear(d_graph, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_cross),
        )
        # Standard Xavier init for last linear; rely on scale=0 in attn proc.
        nn.init.xavier_uniform_(self.proj[-1].weight, gain=0.1)  # small gain for stability
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, node_tokens: Tensor) -> Tensor:
        return self.proj(self.norm(node_tokens))


# ---------------------------------------------------------------------------
# 2. Graph-augmented attention processor (drop-in for SDXL cross-attn)
# ---------------------------------------------------------------------------

class GraphAttnProcessor(nn.Module):
    """
    Replaces a cross-attention processor in the SDXL UNet.

    Maintains a runtime registry pointer to the current batch's graph_tokens
    via `set_graph_tokens(...)` (called once per UNet forward).

    Notes
    -----
    - For SELF-attention layers (encoder_hidden_states is None or equal to
      hidden_states), this processor falls back to the standard path.
    - For CROSS-attention, we add the parallel graph attention.
    - We re-use the UNet's own `to_q` for query (shared with text path),
      so graph attention uses the same query feature; only K, V differ.
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int,
                 n_heads: int = 8) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.n_heads = n_heads

        self.to_k_graph = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_graph = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.zeros(1))    # zero-init residual scale

        # Runtime: external registry sets this each forward
        self._graph_tokens: Optional[Tensor] = None

    def set_graph_tokens(self, graph_tokens: Optional[Tensor]) -> None:
        """Called once per UNet forward. None → text-only path."""
        self._graph_tokens = graph_tokens

    def __call__(
        self,
        attn,                                     # diffusers.Attention
        hidden_states: Tensor,                    # [B, L, D]
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        temb: Optional[Tensor] = None,
        *args, **kwargs,
    ) -> Tensor:
        residual = hidden_states

        # ---- Standard text-attention path (SDXL default) ------------------
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states                  # self-attn
            is_cross = False
        else:
            is_cross = True

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states)
        v = attn.to_v(encoder_hidden_states)

        B, L, D = q.shape
        head_dim = D // attn.heads
        q_view = lambda t: t.view(B, -1, attn.heads, head_dim).transpose(1, 2)
        Q  = q_view(q)
        Kt = q_view(k)
        Vt = q_view(v)

        text_out = F.scaled_dot_product_attention(
            Q, Kt, Vt, attn_mask=None, dropout_p=0.0,
        )                                                          # [B, h, L, hd]

        # ---- Parallel graph-attention path (cross-attn only) --------------
        if is_cross and self._graph_tokens is not None:
            gt = self._graph_tokens
            # Handle CFG batch (uncond+cond stacked) — tile if needed
            if gt.size(0) != B:
                gt = gt.repeat(B // gt.size(0), 1, 1)
            Kg = q_view(self.to_k_graph(gt))
            Vg = q_view(self.to_v_graph(gt))
            graph_out = F.scaled_dot_product_attention(
                Q, Kg, Vg, attn_mask=None, dropout_p=0.0,
            )
            out = text_out + self.scale * graph_out
        else:
            out = text_out

        out = out.transpose(1, 2).reshape(B, L, D)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        if attn.residual_connection:
            out = out + residual
        return out / attn.rescale_output_factor


# ---------------------------------------------------------------------------
# 3. Top-level adapter that wires everything to a UNet
# ---------------------------------------------------------------------------

class SDXLGraphAdapter(nn.Module):
    """
    Top-level container managing:
      - GraphProjector (TGNN → cross-attn dim)
      - All GraphAttnProcessor instances (one per cross-attn layer)
      - install/uninstall to UNet, set/clear graph tokens

    Usage
    -----
        adapter = SDXLGraphAdapter(d_graph=256, cross_dim=2048).to(device, dtype)
        adapter.install(unet)              # replace cross-attn processors
        ...
        graph_tokens = adapter.encode(node_tokens)   # [B, N_g, 2048]
        adapter.set_graph_tokens(graph_tokens)
        out = unet(noisy, t, prompt_embeds, ...)
        adapter.clear_graph_tokens()
    """

    def __init__(self, d_graph: int = 256, cross_dim: int = 2048) -> None:
        super().__init__()
        self.projector  = GraphProjector(d_graph=d_graph, d_cross=cross_dim)
        self.cross_dim  = cross_dim
        # Processors are added in install()
        self._processors: Dict[str, GraphAttnProcessor] = {}

    # ------------------------------------------------------------------
    def install(self, unet) -> None:
        """
        Replace every CROSS-attention processor in unet with GraphAttnProcessor.
        Self-attention processors are kept as default.

        Per-hidden-size sharing: SDXL has cross-attn layers at hidden sizes
        320, 640, 1280. We create ONE pair of K/V graph projectors per size
        and reuse it across layers. Drops adapter from ~340M to ~30M params.
        """
        # Create one shared (to_k_graph, to_v_graph) per (hidden, cross) signature
        sig_groups: Dict[tuple, "GraphAttnProcessor"] = {}
        new_procs = {}
        for name in unet.attn_processors.keys():
            if not name.endswith("attn2.processor"):
                new_procs[name] = unet.attn_processors[name]
                continue
            module_name = name.replace(".processor", "")
            module = unet
            for part in module_name.split("."):
                module = getattr(module, part)
            hidden_size = module.to_q.in_features
            cross_dim   = module.to_k.in_features
            sig = (hidden_size, cross_dim)
            if sig not in sig_groups:
                sig_groups[sig] = GraphAttnProcessor(
                    hidden_size=hidden_size, cross_attention_dim=cross_dim,
                )
            proc = sig_groups[sig]
            new_procs[name] = proc
            self._processors[name] = proc

        unet.set_attn_processor(new_procs)

        # Register *unique* processors as submodules so .parameters() picks them up
        for i, ((sig, p)) in enumerate(sig_groups.items()):
            self.add_module(f"proc_h{sig[0]}_c{sig[1]}", p)

        # Move new processors to same device/dtype as the projector (already cast)
        target_device = self.projector.proj[-1].weight.device
        target_dtype  = self.projector.proj[-1].weight.dtype
        for p in sig_groups.values():
            p.to(device=target_device, dtype=target_dtype)

        print(f"  Installed GraphAttnProcessor on {len(self._processors)} layers "
              f"({len(sig_groups)} unique signatures)")

    # ------------------------------------------------------------------
    def encode(self, node_tokens: Tensor) -> Tensor:
        """[B, N_g, 256] → [B, N_g, cross_dim]"""
        return self.projector(node_tokens)

    def set_graph_tokens(self, graph_tokens: Optional[Tensor]) -> None:
        for p in self._processors.values():
            p.set_graph_tokens(graph_tokens)

    def clear_graph_tokens(self) -> None:
        self.set_graph_tokens(None)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 4. Pad helper — temporal-aware sampling instead of dumb truncate
# ---------------------------------------------------------------------------

def pad_node_embeddings_temporal(
    node_emb: Tensor,        # [N_total, d]
    batch_idx: Tensor,       # [N_total]
    node_frame: Tensor,      # [N_total]    each node's frame index
    target_frame_idx: Tensor,# [B]          target frame index per graph
    is_target_node: Tensor,  # [N_total]    bool, computed by TGNN forward
    n_tokens: int = 64,
    d_graph: int = 256,
) -> Tensor:
    """
    Build per-graph fixed-length token bag with temporal-aware sampling.

    Strategy:
      1. Always keep ALL target-frame nodes (highest priority).
      2. If room remains, fill with non-target nodes ordered by
         |node_frame - target_frame| ascending — i.e. closer frames first.
      3. If still room, zero-pad.
      4. If too many target-frame nodes (> n_tokens), keep first n_tokens
         target nodes (rare; AG average is ~5 nodes per frame).
    """
    device = node_emb.device
    B = int(batch_idx.max().item()) + 1
    out = torch.zeros(B, n_tokens, d_graph, device=device, dtype=node_emb.dtype)

    for i in range(B):
        sel = (batch_idx == i)
        emb_i      = node_emb[sel]
        frame_i    = node_frame[sel].to(device)
        is_tgt_i   = is_target_node[sel].to(device)
        tgt_frame  = target_frame_idx[i].item() if torch.is_tensor(target_frame_idx) else target_frame_idx[i]

        # priority: target-frame nodes first, then by |frame - target|
        dist  = (frame_i - tgt_frame).abs()
        # boolean is_tgt is the strongest sort key
        priority = (~is_tgt_i).long() * 10000 + dist
        order = torch.argsort(priority)              # ascending → target first
        sel_emb = emb_i[order][:n_tokens]
        out[i, :sel_emb.size(0)] = sel_emb
    return out
