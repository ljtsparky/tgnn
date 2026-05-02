# 项目探索全程报告

**项目**：Graph-Conditioned Video Diffusion: Temporal GNN Adapters  
**课程**：UIUC CS598  
**作者**：Jiatong  
**撰写日期**：2026-04-25  

---

## ⚠️ 最大教训：降低分辨率是完全错误的决策

**这是本项目最严重、最愚蠢的失误，直接导致所有训练和生成结果无效。**

### 我们做了什么

为了解决 VAE encode 在 480×720 × 49 帧时的显存不足（OOM），我们把训练和推理分辨率从模型设计的 **480×720 降低到了 256×256**，并在 `run_train.sh` 里写死：

```bash
--frame_h 256 --frame_w 256
```

### 为什么这是完全错误的

CogVideoX-2b 是基于 Patch-based Diffusion Transformer 设计的，原始分辨率为 **480×720**。

在这个分辨率下，VAE（8× 空间压缩）得到 60×90 的 latent，再经过 patch_size=2 的 patch embedding 得到 **1350 个 video patches**，transformer 就是在这 1350 个 token 上进行注意力计算的。

降到 256×256 之后：
- latent 变成 32×32
- video patches 变成 **256 个**（仅为原来的 1/5）
- transformer 在完全不同的 token 数量、不同的空间布局上运行
- 位置编码完全偏离预训练分布
- 模型从未在这个分辨率下训练过，**生成结果退化为噪声/色块**

### 为什么我们能在代码层面改分辨率

**模型权重没有变化**。CogVideoX 的 transformer 理论上支持可变分辨率（patch embedding 可以动态处理不同大小的 latent），所以代码不报错，pipeline 可以运行。但"能运行"不等于"能生成有意义的内容"。我们只是在改变输入帧的大小，让 VAE 产生更小的 latent，然后把这个 latent 塞给从未在此分辨率下见过数据的 transformer。

### 正确的解决方案

OOM 问题不应该通过降分辨率来解决，而应该：
1. **申请更多 GPU**：Delta 的 `gpuA100x4` 分区每节点有 4 张 A100（共 160GB），`--gpus-per-node=4` 即可
2. **使用 `vae.enable_slicing()` + `vae.enable_tiling()`**：在单张 A100 上处理高分辨率，这两个 flag 本来就是为此设计的
3. **申请 `gpuA100x8` 分区**：8 张 A100 共 320GB 显存，完全无压力

### 影响范围

所有在 256×256 下训练的 checkpoint（FiLM、cross-attention）和生成的评测视频**全部需要重新跑**。这几乎等于把之前所有工作推倒重来。

---

## 一、项目目标

给定一张时空场景图（Action Genome 标注：谁在哪、拿什么、做什么、跨帧时序关系），生成一段与该图结构语义一致的室内场景视频。核心问题：如何把结构化图信息注入预训练视频扩散模型？

---

## 二、环境与数据

### 2.1 HPC 环境

| 项目 | 值 |
|---|---|
| 集群 | NCSA Delta |
| GPU | A100 40GB（gpuA100x4 分区）|
| Conda 环境 | /work/nvme/bgnv/leatherman/miniconda3/envs/graphML_tgnn |
| Python / PyTorch / CUDA | 3.11 / 2.6 / cu126 |
| HF 缓存 | /projects/bgnv/leatherman/hf_cache |

### 2.2 数据集：Action Genome + Charades

Action Genome 在 Charades 视频（9601 个）上提供帧级别的时空场景图标注。

```
dataset/ag/
├── annotations/
│   ├── object_classes.txt       # 36 行，第 0 行 __background__，1-35 为物体
│   ├── person_bbox.json         # 每视频每帧的 person bbox + 分辨率
│   └── object_bbox_and_relationship.json  # object bbox + 关系
├── frames/                      # 已 dump 的标注帧 PNG
│   └── VIDEOID.mp4/000001.png ...
└── graphs/                      # 构建完的 PyG 图对象 .pt
```

**踩坑1**：`object_classes.txt` 第 0 行是 `__background__`，node_class 直接用行号，不需要偏移。最初写成 `raw = cls_idx - 1` 导致类别名全部错位。

---

## 三、模块构建历程

### 3.1 场景图构建（ag_graph_dataset.py）

**功能**：把 Action Genome 帧级标注转换成 PyG `Data` 对象。

**节点**（每个物体实例一个节点）：
- `node_class`：物体类别整数（行号即索引）
- `node_bbox`：归一化 bbox [x_center, y_center, w, h]（用帧分辨率归一化）
- `node_frame`：节点所在帧的索引

**边**（28 维 edge_attr）：
- 空间边（同帧 person → object）：attention[4] + spatial[6] + contacting[17] + IoU[1]
- 时间边（跨帧 IoU > 0.3 的同类物体）：同上，IoU 替换为 frame_gap

**运行结果**：CPU job，耗时 8 分钟，生成 9176 个有效图。

---

### 3.2 TGNNEncoder（tgnn_model.py）

**架构**：

```
节点输入：
  class_embed(36 → 256) + bbox_proj(4 → 256) + frame_embed(200 → 256)
  → LayerNorm

边特征：
  edge_proj(28 → 256) + etype_embed(2 → 256)

3 × GATv2Conv(256, heads=8, edge_dim=256, concat=True)
  + pre-norm 残差连接

AttentionalAggregation 池化 → graph_emb [B, 256]

输出：(node_emb [N_total, 256], graph_emb [B, 256])
参数量：796,417
```

**为什么选 GATv2 而不是原版 GAT**：原版 GAT 有 rank-1 bottleneck（注意力分数不依赖 query 节点），GATv2 修复了这个问题，且原生支持 edge_dim 接收边特征。

**踩坑2**：`from torch_geometric.nn import GlobalAttention` 已废弃，需改为 `from torch_geometric.nn.aggr import AttentionalAggregation`。

---

### 3.3 Adapter v1：Token Prepend（失败）

**设计思路**：把 graph_emb 映射到 4 个 token，拼到 text_emb 前面：

```python
graph_tokens = MLP(graph_emb)          # [B, 4, 4096]
cond = cat([graph_tokens, text_emb])   # [B, 230, 4096]
transformer(encoder_hidden_states=cond)
```

**为什么认为可行**：CogVideoX 的 `CogVideoXBlock` 内部用 `text_seq_length = encoder_hidden_states.size(1)` 动态计算，理论上能处理可变长度。

**实际报错**：

```
RuntimeError: The size of tensor a (3558) must match the size of tensor b (3554)
```

**根因**：`CogVideoPatchEmbed` 在 patch_embed 阶段把文本和视频 token 拼接后加位置编码，位置编码是固定大小 `max_text_seq_length(226) + num_video_patches(3328) = 3554`。我们拼了 4 个额外 token，总长变成 3558，不匹配。

**教训**：CogVideoX 的动态 text_seq_length 只在 transformer block 内部生效，patch_embed 层是硬编码的。

---

### 3.4 Adapter v2：FiLM 条件注入

**设计思路**：改用 Feature-wise Linear Modulation，不改变序列长度：

```python
scale, shift = MLP(graph_emb).chunk(2, dim=-1)   # each [B, 4096]
cond = text_emb * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # [B, 226, 4096]
```

序列长度保持 226，不触碰 patch_embed 的位置编码。零初始化最后一层，保证训练初期等价 text-only 基线。

**参数量**：8.66M（TGNN + FiLM adapter 合计 9.5M）

**训练**：256×256，49帧，5000步，batch=1，grad_accum=4，lr=1e-4

**训练结果**：Loss 从 1.85 降至 0.24（5000步）

**踩坑3（gradient checkpointing）**：原本在 transformer forward 之后立刻调 `clear_graph_tokens()`，导致 backward 中 gradient checkpointing 的 recompute 阶段 hooks 无法触发，保存的 tensor 数量不匹配（72 vs 52）崩溃。修复：把 `clear_graph_tokens()` 移到 `loss.backward()` 之后。

**CLIP Score 评估结果**（10个测试视频）：

| 方法 | Mean CLIP |
|---|---|
| FiLM graph-cond | 0.2249 |
| Text-only 基线 | 0.2420 |

Graph-cond 2/10 胜，CLIP score 比 text-only 低。

---

### 3.5 Adapter v3：Cross-Attention Injection（IP-Adapter 风格）

**为什么切换**：FiLM 用单个全局向量 graph_emb 调制所有 text token，信息密度低。IP-Adapter 风格的 cross-attention 让每个 video patch token 直接 attend 到每个图节点 token，保留完整节点级结构。

**架构设计**：

```
TGNNEncoder → node_emb [N_total, 256]
            ↓ pad_node_embeddings(n_tokens=32)
            graph_tokens [B, 32, 256]

在每个 CogVideoXBlock（共 30 个）通过 forward hook 注入：
  video_patches [B, T·H·W, 1920]  ← Query
      × graph_tokens K, V          ← Key/Value（shared projectors: 256→256）
  → cross-attn output              ← 零初始化 to_out，初始不影响生成
  → video_patches += output

不修改 encoder_hidden_states，text_emb 保持 226 token 原样输入。
```

**参数量**：
- TGNNEncoder：796K
- Shared K/V projectors：131K
- 30 × cross-attn block（LayerNorm + to_q + to_out）：29.6M
- 合计：**30.5M**

**Hook 实现关键**：使用 `block.register_forward_hook()` 拦截每个 `CogVideoXBlock` 的输出 `(video_hs, text_hs)`，只修改 `video_hs`，`text_hs` 不动。

**CFG 自动适配**：pipeline 推理时 guidance_scale>1 会把 cond+uncond 拼成 B=2 的批次。Hook 里检测 batch size 不匹配时自动 tile graph_tokens，保证两个 pass 都有图条件。

**训练结果**：Loss 从 1.83 降至 0.006（5000步），比 FiLM 收敛快得多（FiLM 同等步数只到 0.24）。

---

## 四、评估方案的演变

### 4.1 第一版：CLIP Score（错误的指标）

最初用 CLIP 余弦相似度衡量"生成帧与 text prompt 的对齐程度"。

**结论**：该指标不适合本任务。原因：
- text prompt 是从 graph 摘要出来的（`graph_to_prompt`），两个模型用同一个 prompt
- CLIP 无法区分图条件是否有效，它只测文本-图片对齐
- 图条件改变了生成内容，与原始文本描述有偏差，CLIP 分数自然下降
- 这是方法论错误，不是模型的问题

### 4.2 第二版：Object Recall（正确的指标）

**设计**：对生成视频的采样帧跑 CLIP zero-shot 检测，检查图中物体是否出现在视频里。

```
Object Recall = |{objects in graph detected in video}| / |{objects in graph}|
```

对比：graph-cond 是否能比 text-only 更好地生成图中指定的物体。

**第一轮评估（带物体名的 prompt）**：

text prompt = `"a person holding book, clothes, towel indoors"`（包含物体名）

结果：text-only 0.98，graph-cond 0.49。

**发现新问题**：text-only 基线用包含物体名的 prompt，T5 直接编码物体名→CogVideoX 生成这些物体，这是作弊。text-only 的高召回来自 prompt，不是模型能力。

### 4.3 第三版：Generic Prompt + Object Recall（最终正确设计）

把 `graph_to_prompt` 改为完全通用描述：`"a person in an indoor scene"`

重新训练模型（5000步）后评估：

| 方法 | Mean Object Recall |
|---|---|
| Graph-cond（cross-attn）| **0.292** |
| Text-only（generic prompt）| **0.950** |

---

## 五、最终结果分析

### 5.1 数字

```
Graph-cond wins: 0/10
Mean delta: -0.658
```

### 5.2 为什么 text-only 召回率仍然高达 0.95

用 generic prompt `"a person in an indoor scene"` 的 text-only 模型，不知道应该生成哪些物体，但召回率仍接近完美。

原因：**CogVideoX 在大规模室内视频上预训练过**，用通用室内 prompt 就会生成"典型室内场景"——椅子、桌子、食物、衣物等是室内场景的先验，自然出现。Object Recall 检测的就是这些常见物体，text-only 因此虚高。

### 5.3 为什么 graph-cond 更差

Graph-cond 干扰了 CogVideoX 的室内先验，但 5000 步训练不足以让 cross-attention adapter 学会用图结构注入特定物体信息。结果是：既破坏了原有质量，又没学会用图。

### 5.4 结论

**当前系统没有达到预期效果。** 核心瓶颈：

1. **训练不足**：30M 参数的 adapter，5000 步有效更新次数只有 1250（batch×grad_accum），远不够从图到视觉内容建立映射。同等规模的 IP-Adapter 用了 100K+ 步。
2. **低分辨率限制**：256×256 生成质量差，CLIP 检测分数天花板低，0.22 的阈值极度敏感。
3. **训练信号弱**：扩散 MSE loss 对"是否包含特定物体"无监督，模型没有直接动力从图里提取物体类别信息。

---

## 六、所有踩过的坑

| # | 问题 | 症状 | 修复 |
|---|---|---|---|
| 1 | node_class off-by-one | 类别名全部错位 | 不减 1，行号即索引 |
| 2 | GlobalAttention 废弃 | ImportError | 改用 AttentionalAggregation |
| 3 | torchvision VideoReader 未编译 | ImportError | 改用 PIL 读 PNG 帧 |
| 4 | VAE OOM（B=2, 480×720） | 显存不足 | enable_slicing + enable_tiling |
| 5 | Token prepend pos_embedding 不匹配 | 3558 ≠ 3554 | 改 FiLM |
| 6 | Gradient checkpointing + hook 冲突 | recompute 时 tensor 数量不匹配 | clear_graph_tokens 移到 backward 之后 |
| 7 | adapter 权重 dtype 不匹配 | fp32 vs fp16 LinearError | adapter.to(dtype) |
| 8 | CLIP Score 评估方向错误 | 与任务目标不对齐 | 改 Object Recall |
| 9 | Object Recall 评估 prompt 含物体名 | text-only 作弊 | 改 generic prompt，重训 |
| 10 | SLURM 控制器短暂宕机 | sbatch 连接失败 | 重试，非代码问题 |

---

## 七、代码文件清单

| 文件 | 功能 |
|---|---|
| `src/ag_graph_dataset.py` | 从 Action Genome 标注构建 PyG 时空场景图 |
| `src/inspect_graphs.py` | 图质量验证，打印节点/边统计 |
| `src/ag_dataloader.py` | 图 + 视频帧的 DataLoader |
| `src/tgnn_model.py` | TGNNEncoder（GATv2Conv × 3 + AttentionalAggregation）|
| `src/adapter.py` | GraphCrossAttnAdapter（IP-Adapter 风格 cross-attention hook）|
| `src/train.py` | 训练主循环（Stage 2：冻结 CogVideoX，训练 TGNN + Adapter）|
| `src/infer_graph.py` | 推理：输入场景图 → 生成视频 |
| `src/eval_clip.py` | CLIP Score 评估（保留作参考，非主要指标）|
| `src/eval_compositional.py` | Object Recall 评估（主要指标）|
| `src/test_train_dryrun.py` | 全流程 dry-run，验证形状 |

| sbatch 脚本 | 功能 |
|---|---|
| `build_graphs.sh` | CPU 图构建任务 |
| `run_test_dryrun.sh` | GPU dry-run |
| `run_train.sh` | GPU 训练（当前：256×256，5000步，generic prompt）|
| `run_eval_clip.sh` | 生成 10 对视频 + CLIP Score |
| `run_eval_compositional.sh` | Object Recall 评估 |
| `run_infer_graph.sh` | 单视频推理 |

---

## 八、Checkpoint 位置

| 路径 | 内容 |
|---|---|
| `checkpoints/step_005000/` | FiLM adapter（已废弃） |
| `checkpoints_xattn/step_005000/` | Cross-attn，specific prompt 训练 |
| `checkpoints_xattn_generic/step_005000/` | Cross-attn，generic prompt 训练（当前最新）|

---

## 九、如果继续做，应该怎么改

**最有效的改进（优先级排序）**：

1. **训练更多步**：至少 50K 步（参考 IP-Adapter），需要约 50 小时 GPU，分 4~5 次 job 用 `--resume` 接力。

2. **加监督信号**：在 diffusion loss 之外加一个辅助 loss——对生成帧做物体检测，用检测分数作为额外梯度信号，直接告诉 adapter"你需要在视频里生成这些物体"。

3. **提高分辨率**：用 480×720 训练，生成质量更高，Object Recall 检测分数天花板更高。代价是每步更慢，需要更多显存。

4. **更好的 graph_to_prompt**：即使用 generic prompt，也可以保留动作关系但不写物体名，比如 `"a person performing an action indoors"`——让 T5 提供场景结构，图提供物体信息。

5. **换更强的物体检测模型**：当前用 CLIP zero-shot，可换 Grounding DINO 或 OWL-ViT，提高检测精度，降低阈值敏感性。

---

## 十、对创新点的反思

**借鉴的部分**：
- Cross-attention 注入：ControlNet / IP-Adapter 的做法
- GATv2Conv：现成 GNN 模块
- Diffusion loss：标准扩散训练目标

**真正的创新组合**：
- 用 **Temporal GNN** 编码带时序边的 **动态场景图**（Action Genome）
- 作为 **视频扩散模型**（不是图片生成）的结构化条件信号
- 没有先前工作同时做"时序图 + GNN + 视频扩散适配"

**现实差距**：架构设计正确，但训练资源（5000步 vs 需要 50K+ 步）和计算规模（256×256 vs 480×720）不够，导致效果未达预期。这是一个在正确方向上的 early-stage 探索，而不是一个完整验证的系统。
