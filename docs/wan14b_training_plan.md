# Wan2.1-14B 训练计划与完整估算

**目标**：用 Wan2.1-T2V-14B + TGNNEncoder + WanGraphAdapterV2 实现图条件视频生成  
**作者**：Jiatong | UIUC CS598 | 2026-04-29

---

## 一、模型架构对比（1.3B vs 14B）

| 参数 | Wan2.1-1.3B（已跑）| **Wan2.1-14B（目标）** |
|---|---|---|
| 参数量 | 1.3B | **14B（10.8×）** |
| Transformer 层数（num_layers）| 30 | **40** |
| Attention heads | 12 | **40** |
| Head dim | 128 | 128 |
| **内部维度 d_inner** | **1536**（12×128）| **5120**（40×128）|
| FFN dim | 8960 | **13824** |
| Text dim（UMT5-XXL 输出）| 4096 | 4096（相同）|
| VAE | AutoencoderKLWan，同款 | 同款 |
| 文本编码器 | UMT5-XXL（4096-dim）| 同款 |
| 推理显存 | ~8GB | **~24GB** |
| 推理速度（A100）| ~4分钟/视频 | **~17分钟/视频** |
| 生成质量 | 较差（定位轻量）| **SOTA，超越商业模型** |

---

## 二、Wan2.1 完整架构说明

### 2.1 整体组件

```
WanPipeline
├── tokenizer          — T5 tokenizer
├── text_encoder       — UMT5-XXL，输出 [B, seq, 4096]
├── vae                — AutoencoderKLWan（3D causal）
│   ├── 时序压缩：4×
│   ├── 空间压缩：8×
│   └── 潜在通道数：16
├── transformer        — WanTransformer3DModel（14B: 40层）
│   ├── condition_embedder  ← 我们的 hook 注入点
│   ├── rope            — 3D 旋转位置编码（时间+空间）
│   ├── patch_embedding — 时空 patch 化，patch_size=(1,2,2)
│   └── blocks[0..39]   — 40 × WanTransformerBlock
└── scheduler          — FlowMatchEulerDiscreteScheduler
```

### 2.2 WanTransformerBlock（每层结构）

每个 WanTransformerBlock 包含：
1. `norm1` → `attn1`（**Self-Attention**，video patches 自注意力）+ 残差
2. `norm2` → `attn2`（**Cross-Attention**，video patches 作为 Q，encoder_hs 作为 K/V）+ 残差
3. `norm3` → `ff`（前馈网络，dim→ffn_dim→dim）+ 残差

Cross-attention 的 K/V 来自 `encoder_hidden_states`（文本 + 我们的图 token）。

### 2.3 condition_embedder（关键注入点）

```python
class WanTimeTextImageEmbedding:
    def forward(timestep, encoder_hidden_states):
        temb = time_embedder(timestep)          # 时序嵌入
        timestep_proj = time_proj(temb)         # 投影，用于 AdaLayerNorm
        encoder_hidden_states = text_embedder(  # T5 → d_inner
            encoder_hidden_states               # [B, 512, 4096] → [B, 512, 5120]
        )
        return temb, timestep_proj, encoder_hidden_states
```

**我们的 hook 在 condition_embedder 输出之后触发**，得到已经投影到 d_inner（5120）空间的 encoder_hidden_states，然后将图 token 拼接进去。

---

## 三、我们改了哪些部分

### 3.1 改动总结

| 模块 | 是否改动 | 说明 |
|---|---|---|
| VAE | ❌ 冻结 | 不变 |
| UMT5 文本编码器 | ❌ 冻结 | 不变 |
| 40× WanTransformerBlock | ❌ 冻结 | 包括所有 cross-attention 层，不动 |
| condition_embedder | ❌ 冻结（但挂 hook）| 注册 forward hook，拦截输出 |
| **TGNNEncoder** | ✅ **可训练** | 新增，编码场景图 |
| **WanGraphAdapterV2** | ✅ **可训练** | 新增，包含 hook 注册 + graph_proj |

**我们不修改任何 Wan2.1 的原始权重。**

### 3.2 注入流程（14B 版本）

```
输入场景图（Action Genome PyG Data）
    ↓ TGNNEncoder (GATv2Conv × 3 + AttentionalAggregation)
node_emb [N_nodes, 256]
    ↓ pad_node_embeddings(n_tokens=32)
node_tokens [B, 32, 256]   ← 32个节点token，每个256维
    ↓ WanGraphAdapterV2.graph_proj  (256→1024→5120, SiLU, zero-init)
graph_tokens [B, 32, 5120]  ← 投影到Wan内部维度

── 每次 denoising step（共50步） ──
transformer.condition_embedder(timestep, text_emb)
    → encoder_hs = text_embedder(text_emb)   # [B, 512, 5120]
    → [HOOK FIRES] cat([graph_tokens, encoder_hs], dim=1)
    → encoder_hs_injected [B, 544, 5120]     # 32图+512文本
    ↓
40 × WanTransformerBlock.cross_attention
    Q: video_patches [B, T·H·W, 5120]
    K, V: encoder_hs_injected [B, 544, 5120]  ← 全部40层都看到图token
```

### 3.3 WanGraphAdapterV2 参数量（14B版本）

```
TGNNEncoder:
  class_embed(36→256):   9,472
  bbox_proj(4→256):      1,024
  frame_embed(200→256):  51,200
  edge_proj(28→256):     7,168
  etype_embed(2→256):      512
  3× GATv2Conv(256,h=8,ed=256): ~700K
  AttentionalAggregation:   ~27K
  子计:                    796,417

WanGraphAdapterV2 (d_inner=5120):
  input_norm(256):           512
  Linear(256→1024):      263,168
  Linear(1024→5120):   5,243,904  ← zero-init
  子计:                  5,507,584

总可训练参数:            6,303,001 (~6.3M)
```

---

## 四、训练数据方案

### 4.1 为什么纯 Action Genome 不够

- 9K 视频，稀疏标注帧（~24帧/视频），不是完整连续视频
- Adapter 在高度受限分布上训练，无法泛化
- 没有"原始数据"锚点 → adapter 输出偏离 Wan 的 conditioning 分布

### 4.2 正确做法：95% 高质量数据 + 5% 图条件数据

| 数据集 | 类型 | 规模 | 适合度 |
|---|---|---|---|
| **Something-Something V2（SSv2）** | 室内人物-物体交互，高质量 | **220K 视频，2-6秒** | ⭐⭐⭐⭐⭐ |
| Action Genome（已有）| 场景图标注 | 9K 视频 | ⭐⭐⭐（图丰富，视频质量中等）|
| Charades（已下载）| 室内活动视频 | 9.8K 视频 | ⭐⭐⭐⭐ |

**推荐混合方案**：

```
训练数据：
  95% → SSv2（220K视频，无图标注，仅 prompt="a person in an indoor scene"）
          训练时 adapter 的 graph_tokens 全设为零（不激活图条件）
   5% → Action Genome（9K视频，有图标注，正常激活图条件）

效果：
  - SSv2 数据防止 adapter 忘记如何生成高质量室内场景
  - AG 数据教 adapter 如何用图控制具体物体
  - 两者结合 = 高质量 + 图条件
```

SSv2 下载：需要在 qualcomm-ai-research.github.io 申请学术访问（免费）。

### 4.3 数据量估算

| | 数量 | 存储需求 |
|---|---|---|
| SSv2 原始视频（.webm）| 220K | ~20GB |
| SSv2 抽帧（17帧/视频，480P PNG）| 220K × 17 = 3.7M帧 | ~74GB |
| AG 已有帧 | 9K × ~24帧 | 已有 |
| **总存储需求** | | **~100GB** |

---

## 五、GPU Hour 完整估算

### 5.1 训练速度估算

基准：1.3B 在单 A100 上，17帧480P，**8.9秒/步**

14B 规模化估算：
- 层数：40/30 = 1.33×
- 内部维度：(5120/1536)² = 11.1×（attention 计算是二次的）
- 实际缩放（frozen backbone，gradient checkpointing）：约 **4-6×**
- 预计：8.9 × 5 = **~45秒/步**（单卡 A100，2 GPUs for memory safety）

| 配置 | 步速 | 单次5000步 | GPU-hours |
|---|---|---|---|
| 1× A100（40GB，紧张）| ~60s/步 | 83小时 | **83 GPU-h** |
| 2× A100（80GB，推荐）| ~45s/步 | 62小时 | **124 GPU-h** |
| 4× A100（160GB，宽裕）| ~30s/步 | 42小时 | **168 GPU-h** |

### 5.2 各阶段 GPU-hour 预算

| 阶段 | GPU-hours | 说明 |
|---|---|---|
| 下载 Wan2.1-14B | ~0（CPU job）| ~28GB，CPU 分区免费 |
| Dry-run 验证 | ~1 GPU-h | 单次形状验证 |
| 小规模测试训练（1K步）| ~25 GPU-h | 快速验证收敛 |
| 正式训练（5K步，2× A100）| ~124 GPU-h | 主要训练 |
| 推理（10个视频对）| ~5 GPU-h | 评测 |
| Object Recall 评测 | ~3 GPU-h | |
| 备用/调参（额外2次）| ~250 GPU-h | |
| **总计** | **~408 GPU-h** | |

### 5.3 现有预算

- bgnv-delta-gpu 总额度：2000 GPU-hours
- 已使用（CogVideoX + Wan1.3B）：约 300 GPU-hours
- **剩余：~1700 GPU-hours**
- 14B 完整方案需要：~408 GPU-hours
- **结论：预算充足，有 4× 余量**

### 5.4 SBATCH 参数（推荐）

```bash
# 14B 训练（2 GPUs for memory, 48h 时限）
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=48:00:00   # 两天，比62小时多余量

# 14B 推理（1 GPU 够用，有 slicing/tiling）
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
```

---

## 六、与 1.3B 版本的关键差异

| 项目 | 1.3B | 14B |
|---|---|---|
| graph_proj 输出维度 | 256→1024→**1536** | 256→1024→**5120** |
| 可训练参数 | ~5.3M | **~6.3M**（多1M）|
| 40层 vs 30层 | 30 层看到图token | **40 层看到图token** |
| 预期视频质量 | 色块/模糊 | 清晰室内场景 |
| 训练显存需求 | ~16GB | **~40-60GB（需2 GPUs）** |

---

## 七、接下来的步骤（按优先级）

1. **下载 Wan2.1-14B-Diffusers**（CPU job，~28GB）
2. **申请 SSv2 数据集**（academic request，免费）
3. **更新 adapter_wan_v2.py**：d_inner 从硬编码 1536 改为从 transformer.config 动态读取
4. **Dry-run 验证**：确认 14B 下形状正确
5. **小规模训练（500步）**：验证速度和 loss 曲线
6. **正式训练（5K步，95%SSv2+5%AG）**
7. **评测**：Object Recall 对比 text-only baseline
