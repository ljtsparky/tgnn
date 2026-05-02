# 数据与训练管线完整报告

> 目的：把"AG标注 → graph.pt → TGNN → adapter → HunyuanVideo"的每一步具体怎么做的写清楚，让你审视每个环节，找出graph cond"对不上"的潜在原因。

---

## 0. 总览（一张图）

```
Action Genome原始pkl标注
      ↓ ag_graph_dataset.py
graphs/<video_id>.pt    (PyG Data, 含node/edge/frame_ids/video_id/split)
      ↓ AGGraphDataset (训练时5%抽取)
(graph_data, frames[17,3,480,848], prompt="a person in an indoor scene")
      ↓ TGNNEncoder
node_emb [N, 256]        (per-node embedding)
      ↓ pad_node_embeddings (固定32)
node_tokens [B, 32, 256]
      ↓ HunyuanGraphAdapter (LayerNorm + MLP 256→1024→4096, zero-init末层)
graph_cond [B, 32, 4096]
      ↓ torch.cat([graph_cond, prompt_embeds], dim=1)
combined_embeds [B, 32+256, 4096]   ← prepend到LLaMA文本embedding
      ↓ HunyuanVideo Transformer (frozen)
v_pred [B, 16, 5, 60, 106]   (flow matching velocity)
```

---

## 1. 原始数据 — Action Genome (AG)

**来源**：AG = Charades子集的scene graph标注，9176个视频带有完整spatio-temporal scene graph。

**两个核心pkl文件**：
- `person_bbox.pkl` — 每个标注帧的person bbox
- `object_bbox_and_relationship.pkl` — 每个标注帧的object bbox + 与person的关系

**关系类别**（每条person→object关系是multi-label multihot）：

| 类型 | 维度 | 类别 |
|------|-----|------|
| **attention** (3) | `looking_at`, `not_looking_at`, `unsure` |
| **spatial** (6) | `in_front_of`, `behind`, `above`, `beneath`, `in`, `on_the_side_of` |
| **contacting** (17) | `holding`, `not_contacting`, `touching`, `sitting_on`, `leaning_on`, `other_relationship`, `standing_on`, `wearing`, `lying_on`, `covered_by`, `carrying`, `drinking_from`, `eating`, `writing_on`, `wiping`, `have_it_on_the_back`, `twisting` |

**物体类别**（36类）：见 `object_classes.txt`，0=person，1-35=具体物体（chair, table, cup, ...）

**关键事实**：14%的帧没有person检测 → **整个帧被跳过**（不伪造person bbox）。

---

## 2. graph.pt 文件构造（`ag_graph_dataset.py`）

每个 `graphs/<video_id>.pt` 是一个 **PyG `Data` 对象**：

```python
Data(
    node_class  = LongTensor [N],         # 0=person, 1-35=object
    node_bbox   = FloatTensor [N, 4],     # (x,y,w,h) 归一化到[0,1]，per-frame分辨率
    node_frame  = LongTensor [N],         # 该节点属于哪一帧（在valid frames中的索引）
    edge_index  = LongTensor [2, E],
    edge_attr   = FloatTensor [E, 28],    # 见下方
    edge_type   = LongTensor [E],         # 0=spatial, 1=temporal
    frame_ids   = LongTensor [T],         # 原始视频中valid帧的真实编号
    num_frames  = int (T),                # valid帧数
    video_id    = str,                    # e.g. "LLTBQ"
    split       = "train" / "test"
)
```

**节点构造**：
- 每个valid帧 → 1个person节点 + K个object节点（该帧标注里出现的）
- 同一物体在不同帧 = **不同的node**（不去重；后用temporal edge串起来）

**边构造**：
- **Spatial edge** (edge_type=0)：`person → object` 在同一帧内
  - `edge_attr` = `[attn_3 | spat_6 | contact_17 | iou=1.0 | gap=0.0]` （28维）
- **Temporal edge** (edge_type=1)：相邻valid帧之间，**同一物体类**且 IoU≥0.3 的物体连一条
  - `edge_attr` = `[zeros(26) | iou_value | frame_gap_normalised]`
  - person跨帧总是连接（不需要IoU匹配）

**示例**（LLTBQ）：
- 4个object类（clothes, door, doorway, towel）→ 单帧最多约5个node
- 总48个node = 跨多帧累积 (~10帧 × 5节点)
- 71条边 = 同帧person→obj + 跨帧object匹配

---

## 3. 训练时数据如何投喂 — 95/5 混合

### 3.1 Text-only（95%, `CharadesTextDataset`）

```python
# Charades 8019个视频（有预提取frames的子集）
prompt = "a person in an indoor scene"   # 固定，不变
graph_data = None
frames = 从 /frames/<vid>.mp4/ 文件夹均匀采样17帧
```

- **graph_data=None** 在训练循环中触发"零node_tokens"路径
- 目的：让adapter学会"没有graph时不破坏HunyuanVideo原有质量"（防catastrophic forgetting）

### 3.2 Graph-conditioned（5%, `AGGraphDataset`）

```python
graph = torch.load(graphs/<vid>.pt)
frames = _load_frames(graph)   # 从graph.frame_ids对应的真实帧读取
prompt = "a person in an indoor scene"   # 同样固定！
```

**`_load_frames`实现**（`dataset_mixed.py` 行 181-200）：

```python
frame_ids = graph.frame_ids.tolist()           # AG标注里的valid帧
if len(frame_ids) > 17:
    step = len(frame_ids) / 17
    frame_ids = [frame_ids[int(i*step)] for i in range(17)]   # 均匀下采样到17帧
# 否则 < 17帧时用最后一帧repeat填充
```

→ 训练时 frames 是 **AG有标注的那些帧的子集**，按时间顺序排好。**Resize 到 480×848**，归一化 `(x-0.5)/0.5` 到 `[-1, 1]`。

### 3.3 训练循环（`train_hunyuan.py`）

```
for step in 1..5000:
    sample = dataloader.next()   # 95%概率: text-only;  5%概率: graph-cond
    
    1) latents = VAE.encode(frames)              # [B, 16, 5, 60, 106]
    2) flow matching:
       sigma = U(0,1)
       noisy = (1-σ)·latents + σ·noise
       v_target = noise - latents
       timesteps = (sigma * 1000).float()
    3) prompt_embeds, pooled, mask = pipe.encode_prompt(prompt)
       # prompt_embeds: [B, 256, 4096], pooled: [B, 768]
    4) if has_graph:
           node_emb = TGNN(graph)                # [N_total, 256]
           node_tokens = pad_to_32(node_emb)     # [B, 32, 256]
       else:
           node_tokens = zeros(B, 32, 256)        # 无graph信号
       combined_embeds, combined_mask = adapter(node_tokens, prompt_embeds, mask)
       # combined_embeds: [B, 32+256, 4096]
    5) v_pred = transformer(noisy, combined_embeds, pooled, timesteps, mask, guidance=6.0)
    6) loss = MSE(v_pred, v_target) / grad_accum
    7) backward / step
```

**关键参数**：
- batch_size=1, grad_accum=4 → effective batch=4
- 5000优化步 = 20000次forward pass
- 5%抽graph → 1029次实际graph forward pass（实测）
- lr=1e-4, AdamW, 仅训练 TGNNEncoder (796K) + HunyuanGraphAdapter (4.5M) = **~5.3M参数**
- HunyuanVideo Transformer (~13B)、VAE、LLaMA、CLIP **全部冻结**

---

## 4. Graph 编码（`TGNNEncoder` + `pad_node_embeddings`）

### 4.1 节点初始embedding

```python
h_node = (
    class_embed[node_class]   # nn.Embedding(36, 256)
  + bbox_proj(node_bbox)      # Linear(4, 256)
  + frame_embed[node_frame]   # nn.Embedding(200, 256)
)
h_node = LayerNorm(h_node)
```

→ 每个节点 = "类别 + 在哪一帧 + 在画面里什么位置" 的合成向量。

### 4.2 边初始embedding

```python
e = edge_proj(edge_attr) + etype_embed[edge_type]
# Linear(28→256) + Embedding(2, 256)
```

→ 边携带 "关系类型multihot + IoU + 时间间隔 + 是spatial还是temporal" 的信息。

### 4.3 GATv2 message passing

```python
for each of 3 layers:
    h = h + dropout(GATv2Conv(LayerNorm(h), edge_index, edge_attr=e))
```

- GATv2 (Brody et al. 2022) — 每条边的attention权重由 `(src, dst, edge)` 三元组决定
- `n_heads=8`, `d_head=32`, `concat=True` → 输出维度仍是256
- **不加自环**（不让节点和自己交互，避免spatial/temporal语义混合）

### 4.4 输出

```python
node_emb  : [N_total, 256]    # per-node embedding
graph_emb : [B, 256]           # AttentionalAggregation 加权和（一个gate网络打分每个节点）
```

**注意**：在我们的pipeline里**只用`node_emb`，没用`graph_emb`**（adapter输入是pad后的per-node tokens）。

### 4.5 pad_node_embeddings — 关键步骤

```python
def pad_node_embeddings(node_emb, batch_idx, n_tokens=32, d_graph=256):
    out = zeros(B, 32, 256)
    for i in range(B):
        nodes = node_emb[batch_idx == i]   # 该graph的所有节点
        n = min(nodes.size(0), 32)
        out[i, :n] = nodes[:n]              # 取前32个
    return out
```

**⚠️ 重大隐患在这里**：

1. **无序截断**：直接 `nodes[:32]`，**没有按frame或重要性排序**。
   - PyG batch里node顺序由 `Batch.from_data_list` 决定，跟graph构造时的append顺序一致
   - 图大时（如U5T4M有271个节点）只保留前32个，丢失75%以上信息
   - 而且这32个可能集中在前几帧，时间维度信息丢失

2. **零padding**：图小时（如VP4OG只有14节点），剩余18个token是**零向量**。adapter会把这些zero投影到4096维，可能产生噪声token。

---

## 5. Adapter 注入（`HunyuanGraphAdapter`）

### 5.1 结构

```python
input_norm  : LayerNorm(256)
graph_proj  : Sequential(
    Linear(256, 1024),
    SiLU(),
    Linear(1024, 4096)         # 末层 zero-init
)
```

参数量：~4.5M

### 5.2 前向

```python
x          = input_norm(node_tokens)           # [B, 32, 256]
graph_cond = graph_proj(x)                     # [B, 32, 4096]   起始为0（zero-init）
combined   = cat([graph_cond, prompt_embeds])  # [B, 32+256, 4096]
mask       = cat([ones(B,32), prompt_mask])    # [B, 32+256]
```

→ **graph token直接prepend到LLaMA文本embedding序列前面**，作为额外的"伪文本token"。

### 5.3 HunyuanVideo Transformer 怎么用

HunyuanVideo是 **MMDiT** 架构：
- 20个 **double-stream** block：text和video tokens分开但互相attend
- 40个 **single-stream** block：text和video tokens拼接成一个序列做self-attn

我们的32个graph token在两种block里都参与了文本-视频的交互。

**zero-init effect**：训练开始时 `graph_cond=0`，所以 `combined ≈ [zeros(32), prompt]`。zero token经过LayerNorm后还是定值，再经过transformer attention的 softmax/scale，对video tokens的影响接近于0但不为0——这意味着模型一开始**几乎看不见graph信息**，需要慢慢学。

---

## 6. 推理（`infer_hunyuan.py`）

```python
1) 加载HunyuanVideo pipeline + 加载checkpoint adapter
2) 选test split的10个graph (固定seed=42 shuffle)
3) prompt_embeds = pipe.encode_prompt("a person in an indoor scene")
4) for each test graph:
       node_emb = TGNN(graph)
       node_tokens = pad(node_emb, 32)
       combined = adapter(node_tokens, prompt_embeds, mask)
       output = pipe(prompt_embeds=combined, ...,
                     num_frames=17, num_inference_steps=30, guidance_scale=6.0)
       export to <video_id>_hunyuan_graph_cond.mp4
```

baseline完全相同，唯一区别是 baseline 直接 `prompt=PROMPT` 进 pipe，**不经过adapter**。

---

## 7. 我怀疑graph cond效果差的原因（按可能性排序）

### 7.1 ❗ Pad/截断丢信息（很可能是主因）

`pad_node_embeddings` 直接取前32个节点，**没有任何排序或采样策略**：
- 大图（U5T4M 271节点 / 41A89 262节点）保留<12%的节点
- 小图（VP4OG 14节点）有18个零token稀释信号
- 节点顺序是构造时append顺序，时间和重要性都没编码进截断逻辑

**改进方向**：
- 按 `node_frame` 做时间均匀采样
- 或先pool按对象类再展开（保证每类至少出现一次）
- 或增大 `n_graph_tokens` 到 64/128
- 或用 `graph_emb`（AttentionalAggregation）作为单独的"summary token"补充

### 7.2 5% 比例太低 + 5000步太少

- 5000优化步 × 5% = 250次有效graph梯度更新
- adapter从zero-init出发，250次更新难以学出强信号
- HunyuanVideo Transformer 13B参数对4.5M的adapter信号天然抑制

**改进方向**：
- 把比例提到 20-30%（牺牲一些text-only保真，但graph signal更强）
- 训练步数翻倍到 10000-15000
- 或 warmup graph_prob：前期5%（先稳住），后期20%（强化learning）

### 7.3 Prompt太"中性"，adapter没东西可以"挂钩"

prompt是 `"a person in an indoor scene"`，**不包含任何具体物体信息**。这是有意的（防止text泄露ground truth），但同时让transformer没有"语义钩子"接住graph token。

HunyuanVideo的cross-attention是Query=video × Key/Value=text；当text几乎是空白叙述时，graph token要单独把整个语义都建出来——但zero-init的adapter本身就是从无到有学的。

**改进方向**：
- prompt 改成 `"a person interacting with objects in an indoor scene"`（提示transformer去关注objects）
- 或 prompt 用占位 `"a person in an indoor scene with [graph]"`，让graph token填补 `[graph]` 位置
- 或在adapter里加一个learnable prefix bias，让graph token从初始就有"我是物体描述"的语义

### 7.4 节点初始embedding信息不足

`TGNNEncoder` 用 `class_embed + bbox_proj + frame_embed` 编码节点。但：
- class_embed 是从0开始学的nn.Embedding（不是预训练）
- 没有object的视觉特征（只有class index + bbox）
- 没有object的语义文本embedding（如把"chair"过CLIP）

→ TGNN学到的node_emb 本质是"36个class id + 几何信息 + 关系拓扑"的混合，**缺乏与LLaMA语义空间天然对齐的内容**。adapter要把它从这个抽象空间映射到LLaMA 4096维的"chair/table/door"语义空间——非常困难。

**改进方向**：
- 用 CLIP 预先编码 36 个类名 → 512 维语义 → 拼到 class_embed 上
- 或直接让 `class_embed` 初始化为 CLIP 类名 embedding（再训练微调）

### 7.5 一对多的过早瓶颈

graph有变长节点，先压到固定32 token，再过 `Linear(256→1024→4096)` 末层 zero-init。
- adapter只有 ~4.5M 参数承载 32个graph token → 4096维语义的桥接
- 相比 IP-Adapter（每个block都注入）我们只在序列开头prepend一次

**改进方向**：
- 改成 IP-Adapter 风格：每个transformer block内独立加一组K/V projector用graph token做cross-attn
- 或在adapter内加几个self-attn层让32个graph token之间先交互

### 7.6 训练数据frames与graph annotation的对齐问题

`_load_frames` 用 `graph.frame_ids` 取真实帧。但：
- graph的node对应 valid 帧（去掉了14%没有person的帧）
- 17帧均匀采样可能跳过了某些节点对应的关键帧
- 训练时模型看到的frames跟graph描述的内容**有时间错位**

**改进方向**：
- 检查训练时frames与graph节点的对齐
- 或干脆只取graph覆盖最密集的连续17帧

---

## 8. 推荐的改进优先级

| 优先级 | 改动 | 预期收益 | 工作量 |
|--------|------|---------|-------|
| 🔥 高 | `pad_node_embeddings` 改用按帧均匀采样 | 大 | 小 |
| 🔥 高 | 提升 `n_graph_tokens` 32 → 64 | 中 | 极小 |
| 🔥 高 | 用 CLIP class name embedding 初始化 `class_embed` | 大 | 小 |
| 🟡 中 | 训练步数 5000 → 10000 + graph_prob 5%→15% | 中 | 重训 |
| 🟡 中 | prompt改成"a person interacting with objects" | 小-中 | 极小 |
| 🟢 低 | adapter改 IP-Adapter 每block注入 | 大 | 大（需要重写） |

---

## 9. 当前 checkpoint 状态

- 已训练 5000 步，checkpoint每500步保存一次：`step_000500` … `step_005000`
- 10个graph-cond视频已生成：`outputs/hunyuan/`
- 10个 baseline 在A100排队中（job 17987541）
- 评估CSV已dump：`outputs/expected_objects.csv`

---

## 10. 你可以审视的"对照表"

打开 `outputs/expected_objects.csv` 同时对照 `outputs/hunyuan/<video_id>_hunyuan_graph_cond.mp4`：

| video_id | 应该看到 | 实际生成里有吗？ |
|----------|---------|----------------|
| LLTBQ | clothes, door, doorway, towel | ? |
| 17P5V | dish, doorway, laptop, table | ? |
| VP4OG | book | ? |
| Y8L60 | door, doorknob, floor | ? |
| SM8Y0 | chair, table（人坐在椅上） | ? |
| U5T4M | chair, food, refrigerator, sandwich, shelf, table | ? |
| WK9HE | broom, door, floor, table, towel, window（人擦桌子） | ? |
| 41A89 | door, floor, food, pillow, refrigerator, shelf | ? |
| 8YD0O | box, chair, food, laptop, sandwich, table（人坐着吃东西） | ? |
| TRVEA | bed, pillow（人坐床上） | ? |

我个人最猜测的根本原因：**§7.1 (pad截断丢信息) + §7.4 (class_embed跟LLaMA语义不对齐) 双重作用**。
