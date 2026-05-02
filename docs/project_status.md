# 项目全程探索记录

**项目名称**：Graph-Conditioned Video Diffusion: Temporal GNN Adapters  
**课程**：UIUC CS598  
**作者**：Jiatong  
**集群**：NCSA Delta  
**最后更新**：2026-04-23

---

## 一、HPC 环境

| 项目 | 值 |
|---|---|
| 集群 | NCSA Delta |
| 用户名 | leatherman |
| Allocation | bgnv（CPU 200h，GPU 2000h）|
| GPU partition | gpuA100x4（A100 40GB × 4） |
| CPU partition | bgnv-delta-cpu |
| Conda env | /work/nvme/bgnv/leatherman/miniconda3/envs/graphML_tgnn |
| Python / PyTorch / CUDA | 3.11 / 2.6 / cu126 |
| HF cache | /projects/bgnv/leatherman/hf_cache |
| 数据根目录 | /projects/bgnv/leatherman/tgnn/ |
| **注意** | `~` 目录超配额，不要写入；HF_HOME 写在 sbatch 脚本里 |

---

## 二、数据集结构（Action Genome + Charades）

```
dataset/ag/
├── annotations/          # Action Genome 标注
│   ├── object_classes.txt       # 36 行，第 0 行=__background__，1-35=物体类别
│   ├── relationship_classes.txt # 关系类型
│   ├── person_bbox.json         # 每视频每帧的 person bbox + 帧分辨率
│   ├── object_bbox_and_relationship.json  # object bbox + 关系标注
│   └── charades_annotation/
│       └── test.txt / train.txt  # Charades 分割
├── videos/               # Charades 480p .mp4（9601 个）
├── frames/               # 已 dump 的标注帧 PNG
│   └── VIDEOID.mp4/
│       ├── 000001.png
│       └── ...
└── graphs/               # PyG 图对象 .pt（9176 个，已构建）
    └── VIDEOID.pt
```

**关键数据格式**：
- `person_bbox.json`：`{ video_id: { frame_id: {"bbox": [x1,y1,x2,y2], "bbox_size": [H, W]} } }`
- `object_bbox_and_relationship.json`：`{ video_id: { frame_id: [ {class, bbox, attention, spatial, contacting} ] } }`
- edge_attr 维度 28：attn[0:4] + spatial[4:10] + contacting[10:27] + iou[27] + gap（temporal only）

**踩坑**：object_classes.txt 第 0 行是 `__background__`，node_class 直接用行号，**不需要偏移**。最初写成 `raw = cls_idx - 1` 导致 off-by-one。

---

## 三、各模块构建历程

### 3.1 图构建：ag_graph_dataset.py

**功能**：读取 Action Genome 标注，为每个视频构建时空场景图，序列化为 PyG `Data` 对象。

**节点特征**：
- `node_class`：物体类别（int）
- `node_bbox`：归一化 bbox [x_center, y_center, w, h]（用帧分辨率归一化）
- `node_frame`：该节点所在帧的索引（int）

**边特征**（edge_attr 28 维）：
- Attention relations（4 维）
- Spatial relations（6 维）
- Contacting relations（17 维，one-hot）
- IoU（1 维，spatial edges）/ frame gap（temporal edges 中替换 IoU）

**边类型**（edge_type）：
- 0 = 空间边（同帧 person → object）
- 1 = 时间边（跨帧同类 object，IoU > 0.3 阈值匹配）

**运行结果**：
- Job 17798766，耗时 ~8 分钟（CPU）
- 输出：9176 个有效图，覆盖 train/val/test 分割
- 验证指标：IoU > 0.3 的时间边占比 98.2%

**代码路径**：`src/ag_graph_dataset.py`

---

### 3.2 图质量验证：inspect_graphs.py

**功能**：随机抽样图对象，打印节点类别、边统计、时间边密度等。

**发现的问题**：
- `node_class_name()` 函数写成 `raw = cls_idx - 1; return NAMES[raw]`
- 因为 `__background__` 在第 0 行，`cls_idx=1` 应该对应 bag，但结果拿到的是 `__background__`
- **修复**：直接 `return _OBJECT_NAMES_RAW[cls_idx]`（行号即索引，不需偏移）

**运行结果**：
- Job 17798773，通过
- 平均每图节点数：~18，时间边：~12，空间边：~8

**代码路径**：`src/inspect_graphs.py`

---

### 3.3 数据加载器：ag_dataloader.py

**功能**：`AGGraphDataset` 按 split 过滤图，配对 PNG 帧目录，返回 `(PyG Data, frames_tensor)` 对。

**关键设计**：
- 帧目录路径：`frames_dir / f"{video_id}.mp4"` → 直接匹配已 dump 的 PNG 文件夹
- `collate_fn` 返回 `(PyG Batch, list of [T_i, C, H, W])` — 帧不 pad（不同视频帧数不同）

**代码路径**：`src/ag_dataloader.py`  
**测试脚本**：`src/test_dataloader.py`，`run_test_dataloader.sh`

---

### 3.4 TGNNEncoder（tgnn_model.py）

**架构**：3 层 GATv2Conv + AttentionalAggregation 池化

```
节点输入：
  class_embed(36, 256) + bbox_proj(4→256) + frame_embed(200, 256)
  → LayerNorm
  
边特征：
  edge_proj(28→256) + etype_embed(2, 256)

3× GATv2Conv(256→256, heads=8, edge_dim=256, concat=True, add_self_loops=False)
+ pre-norm residual

AttentionalAggregation pooling → graph_emb [B, 256]

输出：(node_emb [N, 256], graph_emb [B, 256])
参数量：796,417
```

**为什么选 GATv2**：
- 原 GAT 有 rank-1 bottleneck（attention score 不依赖 query），GATv2 修复这一问题
- GATv2 原生支持 edge_dim，直接接受边特征而不需要手工拼接

**踩坑**：
- `from torch_geometric.nn import GlobalAttention` 已 deprecated
- 修复：`from torch_geometric.nn.aggr import AttentionalAggregation`
- 参数量检测阈值设太高（> 1M）导致测试失败，改为 > 500K

**代码路径**：`src/tgnn_model.py`  
**测试脚本**：`src/test_tgnn_model.py`，`run_test_tgnn.sh`

---

### 3.5 GraphCondAdapter（adapter.py）— 两个版本

#### v1：Token Prepend（失败）

**设计**：
```
graph_emb [B, 256]
  → LayerNorm → Linear(256, 1024) + SiLU → Linear(1024, 4*4096)
  → reshape → graph_tokens [B, 4, 4096]
  → cat([graph_tokens, text_emb]) → cond [B, 230, 4096]
```

**失败原因**（job 17806234，step 7 崩溃）：
```
RuntimeError: The size of tensor a (3558) must match the size of tensor b (3554) at non-singleton dimension 1
```

**根因分析**：
CogVideoX 的 `CogVideoPatchEmbed` 在 forward 中：
```python
joint_pos_embedding = zeros(1, max_text_seq_length(226) + num_patches(3328), embed_dim)
joint_pos_embedding[:, 226:] = 3D sincos video embeddings
# text 部分全零（text 不需要绝对位置编码）
```
pos_embedding 固定大小 3554。拼接 4 个 graph token 后，total = 230+3328 = 3558，尺寸不匹配崩溃。

注意：CogVideoXBlock 内部用 `text_seq_length = encoder_hidden_states.size(1)` 动态计算，**可以**处理可变序列长度。但 patch_embed 层的 pos_embedding 是固定的——这是 bug 的来源。

#### v2：FiLM 条件注入（当前版本）

**设计**：
```
graph_emb [B, 256]
  → LayerNorm
  → Linear(256, 1024) + SiLU
  → Linear(1024, 2*4096)  ← 零初始化
  → chunk → scale [B, 4096], shift [B, 4096]

cond = text_emb * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
     # [B, 226, 4096]  ← 序列长度不变
```

**FiLM 的优势**：
- 序列长度保持 226，不碰 pos_embedding ✓
- 零初始化 → 初始时 scale=0, shift=0 → 等价于 text-only 基线 ✓
- 每个 token 都受到图结构的调制，而不是只有前几个位置 ✓
- 参数量约 8.66M（比 v1 的 17M 更小）

**代码路径**：`src/adapter.py`  
**测试脚本**：`src/test_adapter.py`，`run_test_adapter.sh`

---

### 3.6 训练流程：train.py

**训练目标**（ε-prediction diffusion loss）：
```
latents = VAE.encode(frames) * scaling_factor    # [B, T_lat, C, H, W]
ε ~ N(0, I)
t ~ Uniform[0, 1000)
x_t = sqrt(ᾱ_t) * latents + sqrt(1-ᾱ_t) * ε   # DDPMScheduler
graph_emb = TGNNEncoder(scene_graph)
cond = GraphCondAdapter(graph_emb, text_emb)      # FiLM [B, 226, 4096]
ε̂ = Transformer(x_t, encoder_hidden_states=cond, t)
loss = MSE(ε̂, ε)
```

**冻结部分**：VAE + T5 text encoder + CogVideoX Transformer（全部参数冻结）  
**训练部分**：TGNNEncoder（796K）+ GraphCondAdapter（8.66M），共约 9.5M 参数

**文本提示构造**（graph_to_prompt）：
```python
"a person {dominant_relation} {object1}, {object2} indoors"
# e.g. "a person not contacting bag, chair, table indoors"
```
- 取 contacting 关系的投票最多项作为 dominant relation
- 仅取前 3 个物体类别

**内存优化**：
- `vae.enable_slicing()` + `vae.enable_tiling()` → 避免 480×720×49 帧 OOM
- `transformer.enable_gradient_checkpointing()` → 节省 adapter 反传的激活内存
- Mixed precision fp16

**训练参数**：
| 参数 | 值 |
|---|---|
| n_steps | 10,000 |
| batch_size | 1 |
| grad_accum | 4（有效 batch=4） |
| lr | 1e-4（linear warmup 200 steps）|
| frame_h × frame_w | 480 × 720 |
| n_frames | 49 |

**代码路径**：`src/train.py`  
**提交脚本**：`run_train.sh`（GPU，12h）

---

## 四、遇到的所有 Bug 与修复

| # | Bug | 症状 | 修复 |
|---|---|---|---|
| 1 | node_class off-by-one | `cls_idx=1` → 显示 `__background__` | `return NAMES[cls_idx]`（不减 1）|
| 2 | GlobalAttention deprecated | ImportError | 改用 `torch_geometric.nn.aggr.AttentionalAggregation` |
| 3 | TGNNEncoder 参数量检查阈值 | `n_params > 1_000_000` fail（796K 是正确的）| 改为 `> 500_000` |
| 4 | VideoReader 未编译 | `torchvision.io.VideoReader` ImportError | 改为从 `frames/` 读 PNG（PIL），不再依赖 ffmpeg |
| 5 | VAE OOM（B=2, 480×720） | 1.32GB 申请失败，显存只剩 1.25GB | `vae.enable_slicing()` + `.enable_tiling()`；dryrun 用 256×256 |
| 6 | pos_embedding size mismatch | 3558 ≠ 3554，token prepend 方案失败 | 改用 FiLM 条件注入，序列长度保持 226 |

---

## 五、Dry-run 状态（test_train_dryrun.py）

| Step | 内容 | 状态 |
|---|---|---|
| 1 | 加载 CogVideoX-2b pipeline（冻结） | ✅ |
| 2 | 加载图 + 49 帧（256×256）| ✅ `frames [1,49,3,256,256]` |
| 3 | VAE encode → latents | ✅ `[1,13,16,32,32]` |
| 4 | DDPMScheduler 加噪 | ✅ |
| 5 | T5 text encode | ✅ `[1,226,4096]` |
| 6 | TGNNEncoder + GraphCondAdapter（FiLM）| ✅（v2，已修复）|
| 7 | Transformer forward | ⏳ job 17822030 运行中 |
| 8 | MSE loss | ⏳ 待验证 |

---

## 六、Job 历史

| Job ID | 内容 | 结果 |
|---|---|---|
| 17798766 | ag_graph_dataset.py 建图 | ✅ 9176 图 |
| 17798773 | inspect_graphs.py | ✅ 图质量正常 |
| 17798888 | test_adapter.py（v1）| ✅ 形状正确 |
| 17799104 | test_train_dryrun（v1）| ❌ VideoReader + OOM |
| 17806234 | test_train_dryrun（v2）| ❌ pos_embedding mismatch |
| 17822030 | test_train_dryrun（v3，FiLM）| ⏳ 运行中 |

---

## 七、下一步

1. **等待 17822030** — 确认 FiLM 干跑全部通过
2. **提交 run_train.sh** — 10000 步正式训练（GPU，12h）
3. **评估指标**：
   - CLIP frame consistency（生成帧与 graph 描述的相似度）
   - 视觉检查：生成视频中的物体关系是否与输入图一致
4. **Ablation**：text-only baseline vs. FiLM graph conditioning
5. **推理脚本**：给定新图 → 生成视频

---

## 八、关键设计决策汇总

| 决策 | 选择 | 原因 |
|---|---|---|
| Video backbone | CogVideoX-2b | 原生支持 49 帧时序，joint attention 方便注入 |
| Graph encoder | GATv2Conv（3 层，8 头）| 修复 GAT rank-1 bottleneck，原生边特征 |
| 图–视频对齐 | 用标注帧 PNG（不读原始 mp4）| torchvision VideoReader 未编译 ffmpeg |
| 条件注入方式 | FiLM（scale+shift）| Token prepend 导致 pos_embedding 尺寸崩溃 |
| 零初始化 | adapter 最后一层全零 | 训练初期等价 text-only，学习稳定 |
| 内存 | slicing + tiling + gradient checkpointing | A100 40GB 跑 480×720×49 帧 |
| 文本提示 | graph_to_prompt：dominant relation + 前 3 物体 | 给 T5 提供场景摘要，图 tokens 提供精确结构 |
