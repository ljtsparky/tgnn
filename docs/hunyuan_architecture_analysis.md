# HunyuanVideo 架构分析与图条件注入方案

**作者**：Jiatong | UIUC CS598 | 2026-04-29

---

## 一、HunyuanVideo 完整架构

### 1.1 整体组件

```
HunyuanVideoPipeline
├── text_encoder      — LLaMA-3-8B（序列编码器）
│     output: [B, seq, 4096]   每个 token 的上下文表示
├── text_encoder_2    — CLIP ViT-L（池化编码器）
│     output: [B, 768]          全局语义向量
├── vae               — AutoencoderKLHunyuanVideo（3D 因果 VAE）
│     时序压缩: 4×     空间压缩: 8×     latent_channels: 16
│     scaling_factor: 0.476986
├── transformer       — HunyuanVideoTransformer3DModel（~13B）
│     num_layers:        20   ← 双流 block（MMDiT 风格）
│     num_single_layers: 40   ← 单流 block（合并后）
│     num_attention_heads: 24
│     attention_head_dim:  128
│     d_inner = 24×128 =   3072
│     text_embed_dim:      4096（与 LLaMA 输出维度一致）
│     guidance_embeds:     True（嵌入式引导，无需双次前向）
└── scheduler         — FlowMatchEulerDiscreteScheduler
```

### 1.2 Transformer 内部结构：双流 → 单流

HunyuanVideo 采用 **MMDiT（Multimodal Diffusion Transformer）** 架构，与 Stable Diffusion 3、Flux 相同，分为两个阶段：

```
输入
  ┌─────────────────────────────────────────────────────┐
  │  video patches  [B, T·H·W, 3072]                    │
  │  text tokens    [B, seq, 3072]  ← context_embedder  │
  └─────────────────────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────────────────────┐
  │  20 × HunyuanVideoTransformerBlock  （双流）         │
  │                                                     │
  │  video stream:  norm → self-attn → ff → residual   │
  │  text  stream:  norm → self-attn → ff → residual   │
  │                                                     │
  │  Cross-stream attention:                            │
  │    Q from video, K/V from text  +                  │
  │    Q from text,  K/V from video                    │
  │  → 文字和视频 token 互相充分交互，双方都被更新        │
  └─────────────────────────────────────────────────────┘
           ↓  text tokens 拼入 video 序列
  ┌─────────────────────────────────────────────────────┐
  │  40 × HunyuanVideoSingleTransformerBlock  （单流）   │
  │                                                     │
  │  merged = cat([video_tokens, text_tokens], dim=1)  │
  │  norm → full-attention(merged) → ff → residual     │
  │  → 所有 token 共同处理                              │
  └─────────────────────────────────────────────────────┘
           ↓
  输出 v_pred [B, C, T_lat, H_lat, W_lat]
```

**与 CogVideoX / Wan 的关键区别**：
| 特性 | CogVideoX | Wan2.1 | **HunyuanVideo** |
|---|---|---|---|
| 架构风格 | Joint Attention | Cross-Attention DiT | **MMDiT（双流+单流）**|
| 文字在哪更新 | 与视频一起（joint）| 不更新（只作 K/V）| **双方互相更新** |
| 文本编码器 | T5-XXL (4096) | UMT5-XXL (4096) | **LLaMA-3-8B (4096) + CLIP (768)** |
| guidance 方式 | CFG（双次前向）| CFG | **嵌入式（单次前向）** |

### 1.3 Latent 维度

对于 480×848，17帧输入：

```
输入帧:   [B, 17, 3, 480, 848]
VAE 时序压缩 4×: (17-1)/4+1 = 5 帧
VAE 空间压缩 8×: 480/8=60, 848/8=106

Latent:  [B, 16, 5, 60, 106]
         (channels=16, T=5, H=60, W=106)

Video patches（patch_size=(1,2,2)）:
  T_patch=1: 5/1=5
  H_patch=2: 60/2=30
  W_patch=2: 106/2=53
  Total patches per frame: 30×53=1590
  Total: 5×1590 = 7950 video tokens
```

---

## 二、我们的图条件注入方案

### 2.1 注入点选择

HunyuanVideo Transformer 的 forward 签名：
```python
transformer(
    hidden_states           = noisy_latents,    # video
    encoder_hidden_states   = prompt_embeds,    # text/graph tokens
    pooled_projections      = pooled_embeds,    # CLIP pooled（timestep条件）
    timestep                = timesteps,         # float [B]，非 long
    encoder_attention_mask  = attention_mask,
    guidance                = guidance,          # 嵌入式 CFG 值，必传
)
```

`encoder_hidden_states` 是 LLaMA 序列输出，传给所有 20 个双流 block 和 40 个单流 block 的 attention。**这是最直接、最有效的注入位置**：所有 60 个 block 都能看到我们的图 token。

### 2.2 注入流程（完整数据流）

```
Action Genome 场景图
    ↓ AGGraphDataset.__getitem__()
PyG Data（node_class, node_bbox, edge_attr, edge_type）
    ↓ TGNNEncoder（3层 GATv2Conv + AttentionalAggregation）
node_emb [N_nodes, 256]      ← 每个节点的嵌入（图中节点数量可变）
    ↓ pad_node_embeddings(n_tokens=32)
node_tokens [B, 32, 256]     ← 截断/补零到固定 32 个节点 token

LLaMA-3-8B(prompt)
    ↓ pipe.encode_prompt()
prompt_embeds [B, 256, 4096] ← 256 个文本 token（max_sequence_length=256）

─────────────────────────────────────────────
node_tokens → HunyuanGraphAdapter.input_norm → LayerNorm(256)
           → graph_proj: Linear(256,1024)+SiLU+Linear(1024,4096,zero-init)
           → graph_cond [B, 32, 4096]

combined_embeds = cat([graph_cond, prompt_embeds], dim=1)  → [B, 288, 4096]
combined_mask   = cat([ones(32), text_mask(256)], dim=1)   → [B, 288]
─────────────────────────────────────────────
    ↓ pipe.transformer(encoder_hidden_states=combined_embeds, ...)

20 双流 block：
  Q=video_patches, K/V=combined_embeds（288 tokens）
  → video 关注 32 个图节点 + 256 个文本 token
  → 图结构信息进入每个 video patch 的特征表示

40 单流 block：
  merged = cat([video_tokens, combined_embeds(288)])
  → 统一 attention，图节点与视频帧深度融合

    ↓ FlowMatchEulerDiscreteScheduler（50步去噪）
生成视频帧 [B, 17, 3, 480, 848]
```

### 2.3 与 CogVideoX / Wan 注入方式的对比

| 特性 | CogVideoX（失败）| Wan2.1（hook）| **HunyuanVideo（直接注入）**|
|---|---|---|---|
| 注入位置 | encoder_hidden_states 前面 | condition_embedder 后 hook | **encoder_hidden_states 前面** |
| 是否需要 hook | 否（但崩了）| 是（较复杂）| **否（最简洁）**|
| 为什么 CogVideoX 失败 | patch_embed 的 pos_embedding 固定长度 3554 | — | **HunyuanVideo 无此限制** |
| pos_embedding 限制 | 是（226+3328=3554 固定）| 无（动态 cross-attn）| **无（context_embedder 动态处理）** |

---

## 三、训练方案

### 3.1 Flow Matching 目标函数

HunyuanVideo 使用 Flow Matching（而非 DDPM）：

```
σ ~ U[0, 1]                         # 均匀采样噪声水平
x_0 = VAE.encode(frames) × 0.476986 # 干净 latent
ε ~ N(0, I)                          # 高斯噪声
x_t = (1-σ) × x_0 + σ × ε           # 线性插值（Flow path）

v* = ε - x_0                        # 速度目标（从数据指向噪声）
v̂  = transformer(x_t, cond, σ×1000) # 预测速度

loss = MSE(v̂, v*)
```

关键：timestep 传 **float**（σ×1000），不是 DDPM 的 long int。

### 3.2 95/5 混合训练

```python
if has_graph:  # 5%
    node_tokens = real_pad_node_embeddings(TGNNEncoder(graph))
else:           # 95%
    node_tokens = torch.zeros(B, 32, 256)  # 零图 token

# 无论哪种情况，都走 adapter（保证梯度路径）
combined_embeds, combined_mask = adapter(node_tokens, prompt_embeds, attn_mask)
```

**为什么 text-only 也要走 adapter**：
`prompt_embeds` 来自 `@torch.no_grad()` 的文本编码，不在计算图上。若 `combined_embeds = prompt_embeds`，则 `v_pred` 没有 grad_fn，`loss.backward()` 崩溃。传零图 token 让 adapter 始终参与前向，梯度路径始终有效。

**零图 token 的语义**：zero-init 的最后线性层保证 `graph_proj(zeros) ≈ 0`（初始时严格为 0，训练后 bias 增长但量值很小）。模型从 zero-injection 学到：当图信息不存在时，adapter 不改变生成。

### 3.3 训练配置

| 参数 | 值 | 说明 |
|---|---|---|
| 冻结 | VAE, LLaMA, CLIP, Transformer | 全部 HunyuanVideo 参数 |
| 可训练 | TGNNEncoder + HunyuanGraphAdapter | 共 **5.26M 参数** |
| GPU | NVIDIA H200（80GB）| A100 40GB 太紧 |
| 分辨率 | 480×848（16:9，480P）| HunyuanVideo 原生分辨率 |
| 帧数 | 17（16N+1 格式，1秒）| 最短合法格式 |
| 批大小 | 1（grad_accum=4 → 有效批=4）| |
| lr | 1e-4（warmup 200步）| |
| 精度 | bf16（Transformer）/ fp16（VAE）| |
| 数据 | 95% Charades + 5% Action Genome | |

---

## 四、理论依据与文献支持

### 4.1 核心方法论：Soft Prompt Prepending

**为什么在 encoder_hidden_states 前面拼接图 token 是有效的？**

直觉：LLaMA 的 `prompt_embeds` 是一个 token 序列，HunyuanVideo 的 attention 对这个序列做的是**全局注意力**（无固定位置偏置）。在序列前面加入额外 token，等价于给模型更多"上下文"可以参考。

文献支持：

**[1] Prefix-Tuning（Li & Liang, 2021, ACL）**
- 最早系统性验证了在 token 序列前面加入可训练"前缀 token"能有效引导语言模型生成
- 证明前缀 token 可以编码任意结构化的条件信息
- 与我们方法完全同构：我们的 graph_cond [32, 4096] 就是"前缀 token"

**[2] IP-Adapter（Ye et al., 2023, ICCV）**
- 提出将图像 token 拼接到文本 token，通过 cross-attention 注入图像条件
- 核心发现：在 UNet 的 cross-attention K/V 端附加图像 token，可以在不修改模型的情况下实现精确的视觉条件控制
- 我们的方法是 IP-Adapter 的时序图版本，同样是"拼接额外 token + 让 attention 自由选择"

**[3] ControlNet（Zhang et al., 2023, ICCV）**
- 提出将额外条件通过冻结主干 + 可训练副本注入，最终 zero-conv 合并
- **Zero-init** 技巧与我们完全相同：初始时注入量=0，模型从文本基线出发学习
- 证明了"zero-init 新分支 + 冻结主干"是稳定微调的最佳实践

**[4] SGDiff（Yang et al., 2023）**
- 首次将静态场景图条件注入扩散模型（图像生成）
- 证明了场景图中的节点/边特征可以被 GNN 编码为有效的扩散条件
- 我们的工作是其视频 + 时序图的自然扩展

**[5] SceneGenie（Farshad et al., 2023, ICCV）**
- 在推理时用场景图引导扩散轨迹（无需微调）
- 验证了场景图信息可以控制扩散模型的生成内容
- 我们的工作把这个想法做成了端到端训练的 adapter

### 4.2 为什么 HunyuanVideo 特别适合我们的方案

**双流 MMDiT 的优势**：

在双流 block 中，video token 和 text token **互相 attend**（不是单向的 cross-attention）。这意味着：
- video patch 可以 attend 到我们的 32 个图 token → 图结构影响视频内容
- 图 token 也可以 attend 到 video patch → 图 token 的语义会被视频上下文精炼

这种**双向交互**比 Wan 的单向 cross-attention 更强，理论上能学到更精确的图-视频对齐。

**无固定序列长度约束**：

不同于 CogVideoX 的 `CogVideoPatchEmbed` 有固定 `max_text_seq_length=226`，HunyuanVideo 的 `context_embedder` 动态处理任意长度的 `encoder_hidden_states`（通过 `encoder_attention_mask` 指定有效长度），完全支持我们的 32+256=288 token 序列。

**[6] Flow Matching 的优势（Lipman et al., 2022; Liu et al., 2022）**
- Flow matching 的训练目标（MSE 在速度场上）比 DDPM 的 ε-prediction 更稳定
- 在 same NFE（推理步数）下，flow matching 通常比 DDPM 产生更高质量的样本
- HunyuanVideo 使用 flow matching 是其高质量的原因之一

### 4.3 我们方案的局限性与改进方向

**现有局限**：
1. 训练步数（5000步）相比 IP-Adapter（100K+ steps）差距很大
2. Action Genome 只有 9K 视频，远小于 IP-Adapter 的训练规模
3. 17 帧（1秒）vs HunyuanVideo 原生 129 帧（8秒），语义信息不充分

**理论上更好的方案**：
- 用 IP-Adapter 的"专用 cross-attention 头"而不是拼接（参数效率更高）
- 用 LoRA 微调 LLaMA 的 K/V 投影，让文本编码器也能理解图结构
- 增加时序对比学习 loss：生成的帧与输入图的场景图匹配度

---

## 五、我们的创新点定位

| 层次 | 借鉴 | 我们的贡献 |
|---|---|---|
| 注入方式 | IP-Adapter（图像条件拼接）| 移植到视频生成 + 时序图 |
| Zero-init | ControlNet | 相同技巧，不同模态 |
| 图编码器 | GATv2Conv（现成模块）| 用 Action Genome 时序边构建时序图 |
| 目标函数 | Flow Matching（HunyuanVideo 标准）| 无修改 |
| **核心创新** | **无先例** | **时序场景图（含跨帧 IoU 匹配边）→ 视频扩散适配**，端到端框架 |

没有先前工作同时做：**时序 GNN + 视频扩散 adapter + Action Genome 时序图**这个完整组合。这是本项目的学术贡献。
