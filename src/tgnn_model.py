"""
tgnn_model.py
=============
Temporal Graph Neural Network encoder for Action Genome scene graphs.

Architecture overview
---------------------
Input : PyG Data (or Batch) with fields from ag_graph_dataset.py:
    node_class  [N]     int64    0=person, 1-35=object class
    node_bbox   [N, 4]  float32  (x,y,w,h) normalised to [0,1]
    node_frame  [N]     int64    frame index (0-indexed, 0..T-1)
    edge_index  [2, E]  int64
    edge_attr   [E, 28] float32  relationship multihot + iou + gap
    edge_type   [E]     int64    0=spatial, 1=temporal
    batch       [N]     int64    (added by PyG Batch; absent for single graphs)

Output:
    node_emb    [N, d_model]   float32   per-node embedding
    graph_emb   [B, d_model]   float32   per-graph embedding (goes to adapter)

Design decisions
----------------
1. GATv2Conv (not GGNN/GRU-based):
   - Accepts edge_dim natively — all 28 dims of edge_attr are used.
   - GATv2 fixes the rank-1 expressiveness bottleneck of original GAT
     (Brody et al., 2022: "How Attentive are Graph Attention Networks?").
   - GGNN would need an MLP producing a d×d message matrix per edge;
     at d=256 that's 65K parameters per edge — prohibitive.

2. Unified spatial + temporal message passing:
   - Both edge types live in one edge_index; the model is not told to
     treat them separately at the conv level. Instead, edge_type (0/1)
     is embedded and *added* to the projected edge_attr before the conv.
   - This lets the attention mechanism learn different aggregation patterns
     for spatial vs temporal edges without hard-wiring the separation.

3. Node input = class_embed + bbox_proj + frame_pos_embed:
   - frame_pos_embed encodes temporal position (0..T-1); without this the
     model cannot distinguish "person in frame 3" from "person in frame 10"
     even if temporal edges exist, because nodes are orderless in GNN input.

4. GlobalAttention pooling (Vinyals et al., "Order Matters"):
   - A learned gate network scores each node; weighted sum → graph_emb.
   - Better than mean/max pool for scene graphs because person and object
     nodes contribute unequally to the global scene description.

5. Pre-norm residual (LayerNorm before conv, not after):
   - More stable gradient flow at the start of training.
   - Matches the convention used in CogVideoX's DiT blocks (the adapter
     target), keeping normalisation styles consistent.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.aggr import AttentionalAggregation


# ---------------------------------------------------------------------------
# Constants — must match ag_graph_dataset.py
# ---------------------------------------------------------------------------

N_OBJ_CLASSES  = 36     # node_class in [0, 35];  0 = person
EDGE_ATTR_DIM  = 28     # edge_attr width
N_EDGE_TYPES   = 2      # 0 = spatial, 1 = temporal
MAX_FRAMES     = 200    # safe upper bound on T per video (dataset max is 126)


# ---------------------------------------------------------------------------
# TGNNEncoder
# ---------------------------------------------------------------------------

class TGNNEncoder(nn.Module):
    """
    Parameters
    ----------
    d_model   : hidden dimension throughout (must be divisible by n_heads)
    n_layers  : number of GATv2 message-passing rounds
    n_heads   : attention heads per GATv2 layer
    dropout   : dropout rate applied after each conv (and in attention)
    """

    def __init__(
        self,
        d_model:  int   = 256,
        n_layers: int   = 3,
        n_heads:  int   = 8,
        dropout:  float = 0.1,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        d_head = d_model // n_heads

        # ── Node input projections ───────────────────────────────────────────
        self.class_embed = nn.Embedding(N_OBJ_CLASSES, d_model)
        self.bbox_proj   = nn.Linear(4, d_model)
        self.frame_embed = nn.Embedding(MAX_FRAMES, d_model)
        # is_target embedding: marks "this node is in the target frame".
        # 0 = non-target, 1 = target. Lets the TGNN distinguish "context" vs
        # "frame to render" during message passing — without this the network
        # would treat every node identically and lose the temporal anchor.
        self.target_embed = nn.Embedding(2, d_model)
        self.input_norm   = nn.LayerNorm(d_model)

        # ── Edge input projections ───────────────────────────────────────────
        self.edge_proj   = nn.Linear(EDGE_ATTR_DIM, d_model)
        self.etype_embed = nn.Embedding(N_EDGE_TYPES, d_model)

        # ── GATv2 message-passing layers ─────────────────────────────────────
        # concat=True → output is n_heads * d_head = d_model  (shape preserved)
        # add_self_loops=False — self-connections are not meaningful for typed
        # scene-graph edges and would mix spatial/temporal semantics.
        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels  = d_model,
                out_channels = d_head,
                heads        = n_heads,
                edge_dim     = d_model,
                concat       = True,
                add_self_loops = False,
                dropout      = dropout,
            )
            for _ in range(n_layers)
        ])
        # Pre-norm: normalise the residual stream *before* each conv
        self.pre_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout   = nn.Dropout(dropout)

        # ── Graph-level pooling ──────────────────────────────────────────────
        # gate_nn  : scores each node (higher → more important to graph summary)
        # nn       : transforms the node feature before weighted sum
        self.graph_pool = AttentionalAggregation(
            gate_nn = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            ),
            nn = nn.Linear(d_model, d_model),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.normal_(self.frame_embed.weight, std=0.02)
        nn.init.normal_(self.target_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.bbox_proj.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)

    def init_class_embed_from_clip(
        self, class_names: list[str], clip_model_name: str = "openai/clip-vit-large-patch14",
    ) -> None:
        """
        One-shot init: replace class_embed with CLIP-text-encoded class names.

        For each AG class name (e.g. 'chair'), encode 'a photo of a {name}' with
        CLIP and project to d_model via the existing class_embed weight matrix
        (Linear-init). This gives the encoder semantic-aligned class priors
        instead of starting from random embeddings.
        """
        from transformers import CLIPModel, CLIPProcessor
        device = self.class_embed.weight.device
        d = self.class_embed.weight.size(1)

        clip = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
        proc = CLIPProcessor.from_pretrained(clip_model_name)
        prompts = [f"a photo of a {n}" for n in class_names]
        with torch.no_grad():
            tok = proc(text=prompts, return_tensors="pt", padding=True).to(device)
            # Some transformers versions return ModelOutput; extract pooler_output
            txt_out = clip.text_model(input_ids=tok["input_ids"],
                                      attention_mask=tok.get("attention_mask"))
            pooled = txt_out.pooler_output              # [C, hidden]
            text_emb = clip.text_projection(pooled)     # [C, projection_dim]
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Project 768 → d_model with a fixed random projection (kept frozen
        # afterwards by overwriting the embedding table directly).
        proj = torch.empty(text_emb.size(1), d, device=device)
        nn.init.xavier_uniform_(proj)
        new_emb = text_emb @ proj                                # [C, d]
        new_emb = new_emb / new_emb.norm(dim=-1, keepdim=True) * 0.5

        n_clip = new_emb.size(0)
        n_table = self.class_embed.weight.size(0)
        with torch.no_grad():
            self.class_embed.weight[:min(n_clip, n_table)].copy_(new_emb[:min(n_clip, n_table)])
        del clip
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def forward(
        self,
        data: Data | Batch,
        target_frame_idx: Tensor | None = None,   # [B] or None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        data : PyG Data or Batch.
        target_frame_idx : [B] long tensor giving the target_frame_idx for
            each graph in the batch (in 0..T_i-1, position within frame_ids).
            None → no target marking (pure context encoding).

        Returns
        -------
        node_emb       : [N, d_model]   per-node embeddings (after MP)
        graph_emb      : [B, d_model]   per-graph embeddings
        is_target_node : [N] bool       True where node is in its graph's target frame
        """
        # ── Node features ────────────────────────────────────────────────────
        frame_idx = data.node_frame.clamp(max=MAX_FRAMES - 1)
        h = (
            self.class_embed(data.node_class)   # [N, d]
            + self.bbox_proj(data.node_bbox)    # [N, d]
            + self.frame_embed(frame_idx)       # [N, d]
        )

        # ── Target-frame marker ──────────────────────────────────────────────
        # Compute is_target per node by comparing data.node_frame with the
        # target_frame_idx of the graph the node belongs to.
        if hasattr(data, "batch") and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = h.new_zeros(h.size(0), dtype=torch.long)
        if target_frame_idx is None:
            is_target_node = torch.zeros(h.size(0), dtype=torch.bool, device=h.device)
        else:
            tgt_per_node = target_frame_idx.to(h.device)[batch_vec]   # [N]
            is_target_node = (data.node_frame == tgt_per_node)         # [N]
        h = h + self.target_embed(is_target_node.long())                # [N, d]
        h = self.input_norm(h)                  # [N, d]

        # ── Edge features ────────────────────────────────────────────────────
        e = (
            self.edge_proj(data.edge_attr)      # [E, d]
            + self.etype_embed(data.edge_type)  # [E, d]
        )                                       # [E, d]

        # ── Message passing (pre-norm residual) ──────────────────────────────
        ei = data.edge_index
        for norm, conv in zip(self.pre_norms, self.convs):
            h = h + self.dropout(conv(norm(h), ei, edge_attr=e))

        # ── Graph-level pooling (post-MP) ────────────────────────────────────
        graph_emb = self.graph_pool(h, batch_vec)   # [B, d]

        return h, graph_emb, is_target_node
