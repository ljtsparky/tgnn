# 项目未来方向：正确做法全记录

**写于**：2026-04-27  
**目的**：记录当前所有技术问题的根因、正确解法、以及通向完整系统的路线图，日后继续推进时直接从这里出发。

---

## 一、当前系统的根本缺陷

### 1.1 adapter 注入位置错误（已修复为 v2）

**问题**：WanGraphAdapter v1 把 graph tokens [B, 32, 4096] 拼到 T5 tokens [B, 512, 4096] 后，作为 `prompt_embeds` 传给 transformer。Wan transformer 内部的 `condition_embedder.text_embedder`（PixArtAlphaTextProjection，4096→1536）会处理这 544 个 token。但我们的 graph tokens **不在 T5 的语义空间里**，text_embedder 对它们的投影产生乱码，导致生成视频是色块。

**正确做法（v2 已实现）**：
- Hook 在 `condition_embedder` 的**输出**上（而不是输入）
- 此时 `encoder_hs` 已经是 1536-dim（text_embedder 之后）
- 把 graph tokens 投影到 **1536-dim**，拼接到 encoder_hs
- 所有 30 个 block 的 cross-attention 就在正确空间里看到图结构

```
T5 tokens [B, 512, 4096]
  → condition_embedder.text_embedder → [B, 512, 1536]
  → hook: cat([graph_proj(graph_tokens), encoder_hs])
  → [B, 32+512, 1536] → 进入 30 个 WanTransformerBlock
```

**代码位置**：`src/adapter_wan_v2.py`

---

### 1.2 训练数据太少（根本瓶颈）

**问题**：只用 Action Genome 的 9K 个室内视频训练 adapter。
- 每个视频只有 ~24 个稀疏标注帧（不是连续视频）
- 种类单一（Charades 室内场景）
- adapter 学到的是 9K 个视频的特定分布，泛化能力极差
- 没有原始 Wan 数据的"锚定"，adapter 容易破坏模型原有质量

---

## 二、正确的混合训练策略

这是学术界处理 adapter/LoRA fine-tuning 的标准做法。

### 2.1 数据构成

```
批次构成：
  90% 大规模视频（Panda-70M / WebVid-10M）
    → 伪场景图（YOLOv8 + 空间关系推算）
    → 质量不高但量大，保持 Wan 原有生成能力
    → 约 500K~5M 视频

  10% Action Genome
    → 人工标注图（gold standard）
    → 质量高，只有 9K 视频
```

### 2.2 Graph Dropout（关键设计）

训练时以概率 `p_drop=0.1` 随机把 `graph_tokens` 设为 `None`（null 图条件）。这使模型学会两种模式：
- 有图：跟随图结构生成
- 无图（null）：回退到文本条件（Wan 原有能力）

这是 Classifier-Free Guidance 的标准做法。推理时：

```python
v_cond  = transformer(noisy, text_emb, graph_tokens=real_graph)
v_uncond = transformer(noisy, text_emb, graph_tokens=None)
v_output = v_uncond + guidance_scale * (v_cond - v_uncond)
```

### 2.3 推荐数据集

| 数据集 | 规模 | 获取方式 | 优势 |
|---|---|---|---|
| **Panda-70M** | 70M clips (~170TB 完整，subset 可下) | [github.com/snap-research/Panda-70M](https://github.com/snap-research/Panda-70M) | 高质量，有 caption |
| **WebVid-10M** | 10M videos | 公开下载（部分链接失效）| 有 text，多样 |
| **HD-VILA-100M** | 100M clips | 需申请 | 高分辨率 |
| **InternVid** | 7.1M clips | HuggingFace 公开 | 质量高，有 caption |

实际操作：下载 InternVid 或 Panda-70M 的 10% subset（~7M clips），用 `build_pseudo_sg.py` 生成伪场景图。

---

## 三、伪场景图生成流程

代码：`src/build_pseudo_sg.py`

```
视频 → 1fps 采样 → YOLOv8 (nano/small) 检测
    → 人物 bbox + 物体 bbox
    → 计算帧内空间关系（上下左右重叠）
    → 计算跨帧时序边（同类物体 + IoU 阈值）
    → 保存为 PyG Data（与 Action Genome 相同格式）
```

**局限性**：伪图质量不如 Action Genome 的人工标注。但对于 adapter 训练而言，"有足够多样的图结构"比"每张图完全准确"更重要。

---

## 四、完整训练 Pipeline（正确版本）

```python
for batch in mixed_dataloader:
    graph, frames, prompts = batch
    
    # graph 可能是:
    #   - Action Genome 人工标注图
    #   - 大规模视频伪标注图
    #   - None（null，概率 p_drop）
    
    latents  = vae.encode(frames)                    # 冻结
    text_emb = t5.encode(prompts)                    # 冻结
    
    node_emb = tgnn(graph)                           # 可训练
    
    if graph is not None and random() > p_drop:
        graph_tokens = pad(node_emb)
        adapter.set_graph_tokens(graph_tokens)       # 可训练
    else:
        adapter.clear_graph_tokens()                 # null: text-only
    
    # Hook 在 condition_embedder 输出上注入
    v_pred = transformer(noisy, text_emb)            # 冻结主干
    adapter.clear_graph_tokens()
    
    loss = flow_matching_loss(v_pred, v_target)
    loss.backward()   # 只更新 tgnn + adapter
```

---

## 五、正确的学术定位

### 我们的贡献是什么

**不是**：更好的场景图检测器（那是 RelTR、DINO-SG 等的工作）

**是**：给定任意来源的场景图（人工或自动），如何将图结构有效注入视频扩散模型

### 伪标签在学术上合理吗

合理。类比：
- ControlNet 用 Canny/HED 从训练图片自动生成边缘图 → 自监督 conditioning
- IP-Adapter 用 CLIP 特征提取器生成图片表示 → 学习 conditioning 机制
- 我们用 YOLO 生成伪场景图 → 学习 graph conditioning 机制

**贡献的核心是 conditioning 机制**，不是标注质量。

### 不应该做的事

- 用 GPT-4V 生成高质量场景图然后说"我们的系统在高质量图上效果好" → 贡献是 GPT-4V，不是我们
- 依赖闭源模型的标注结果作为训练数据（reproducibility 问题）

---

## 六、下一步执行计划

**阶段1：修复注入机制（1-2天）**
- [x] 写 `adapter_wan_v2.py`（hook 在 condition_embedder 输出上）
- [ ] 跑 dry-run 验证 v2 注入正确，生成视频不是色块
- [ ] 用 v2 重训 5000 步，对比 v1 视觉质量

**阶段2：数据扩展（1周）**
- [ ] 下载 InternVid subset（~1M clips，约 50GB）
- [ ] 运行 `build_pseudo_sg.py` 生成伪图
- [ ] 写混合 DataLoader（支持 AG + 大规模视频 + null 图）

**阶段3：大规模混合训练（GPU 密集）**
- [ ] 混合数据：90% 大规模 + 10% AG，p_drop=0.1
- [ ] 使用 gpuA100x4，4 张卡并行（DDP）
- [ ] 训练 20K 步（估计 ~30 小时，分 3 个 12h job）

**阶段4：评测**
- [ ] Object Recall（当前已有）
- [ ] 视觉对比（人眼 + 视频质量评分）
- [ ] Ablation：v1 vs v2 adapter，AG-only vs Mixed 数据

---

## 七、预期效果

修复注入机制后，即使还是 9K AG 数据：
- 视频不应该再出现色块（核心 bug 修复）
- Object Recall 应有改善

加入大规模数据后：
- 视频质量更接近原始 Wan2.1（不退化）
- Graph conditioning 更准确（更多多样性的图结构训练）
- 可以期望 graph-cond > text-only on rare objects（broom, vacuum, medicine 等）
