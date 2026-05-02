# NCSA Delta GPU 资源申请指南

> 数据来源：`sinfo -o "%P %G %m %c %D"` + `scontrol show partition`，2026-04-25

---

## 零、CogVideoX-2b 完整性能显存需求（详细计算）

### 推理显存（官方数据）

| 场景 | 显存需求 | 备注 |
|---|---|---|
| 纯推理 BF16/FP16 | **~5 GB** | 官方最低值 |
| 推理 INT8 量化 | ~4.4 GB | 需要 bitsandbytes |
| 推理 480×720×49帧 | **~16 GB** | 实际测量（无量化，无 slicing）|
| 推理 480×720×49帧 + slicing + tiling | **~10 GB** | 启用内存优化后 |

### 训练显存（详细推导，针对我们的 Stage 2）

**设定**：CogVideoX-2b，FP16，batch=1，480×720，49帧，gradient checkpointing ON

#### 1. 冻结主干模型（只占显存，不产生梯度）

```
Transformer 权重（2B 参数，fp16）：2,000M × 2B = 4.0 GB
VAE 权重（~83M 参数，fp16）：   83M × 2B  = 0.17 GB
T5 文本编码器（~4.7B，fp16）：  4,700M × 2B = 9.4 GB
总主干模型权重：                          ≈ 13.6 GB
```

> 注意：T5-XXL 是显存大户。推理时可 offload 到 CPU（encode 完就不再需要）。

#### 2. 可训练模块（GraphCrossAttnAdapter + TGNNEncoder）

```
参数量：30.5M（fp16 存储）
权重：  30.5M × 2B              = 61 MB
梯度：  30.5M × 4B（fp32）      = 122 MB
优化器状态（AdamW，fp32 m+v）：30.5M × 8B = 244 MB
小计：                                     ≈ 0.43 GB
```

#### 3. 激活值（480×720×49帧，gradient checkpointing）

```
输入帧（fp16）：
  [1, 49, 3, 480, 720] × 2B = 100 MB

VAE latent（fp16）：
  [1, 13, 16, 60, 90] × 2B  = 22 MB

video patches（进 transformer）：
  T_lat=13, H_lat=30, W_lat=45 → 13×30×45 = 17,550 patches
  [1, 17550, 1920] × 2B = 67 MB

Gradient checkpointing 节省约 80% 激活显存（只保存每个 block 的输入）：
  30 blocks × ~30 MB × 20% = 180 MB 激活

Cross-attn hook 的 K/V（graph tokens）：
  [1, 32, 256] × 2B × 30 blocks ≈ 2 MB

总激活估计：                              ≈ 0.37 GB
```

#### 4. 汇总

| 组成 | 显存 |
|---|---|
| 主干模型权重（冻结）| ~13.6 GB |
| T5 offload 到 CPU 后 | ~4.2 GB |
| 可训练 adapter + 梯度 + 优化器 | ~0.43 GB |
| 激活值（gradient checkpointing）| ~0.37 GB |
| CUDA 碎片 + overhead | ~1 GB |
| **总计（T5 offload 后）** | **≈ 6 GB** |
| **总计（T5 不 offload）** | **≈ 16 GB** |

**结论：单张 A100（40GB）完全够用，T5 offload 后约 6-16GB，远未打满。**  
OOM 问题的真正来源是 VAE encode 时的临时峰值（49帧 × 480×720 的中间特征），启用 `enable_slicing()` + `enable_tiling()` 后可控制在 2GB 以内。

**我们之前降到 256×256 完全是没有必要的错误操作。**

---

## 一、GPU 分区总览

| 分区名 | GPU 型号 | 每节点 GPU 数 | 每 GPU 显存 | 每节点 CPU | 每节点 RAM | 节点数 | 最长时限 |
|---|---|---|---|---|---|---|---|
| `gpuA100x4` | NVIDIA A100 | **4** | 40 GB | 64 | 256 GB | 99 | 2 天 |
| `gpuA100x8` | NVIDIA A100 | **8** | 40 GB | 128 | 2 TB | 6 | 2 天 |
| `gpuA40x4` | NVIDIA A40 | **4** | 48 GB | 64 | 256 GB | 98 | 2 天 |
| `gpuH200x8` | NVIDIA H200 | **8** | 80 GB | 96 | 2 TB | 8 | 2 天 |
| `gpuA100x4-interactive` | NVIDIA A100 | 4 | 40 GB | 64 | 256 GB | 100 | 1 小时 |
| `gpuA100x8-interactive` | NVIDIA A100 | 8 | 40 GB | 128 | 2 TB | 6 | 1 小时 |

> `gpuA100x4` 是默认（`Default=YES`）GPU 分区。

---

## 二、`--gpus-per-node` 设置规则

### gpuA100x4 分区（每节点最多 4 张）

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1    # 1 张 A100，40GB
#SBATCH --gpus-per-node=2    # 2 张 A100，80GB 总显存
#SBATCH --gpus-per-node=4    # 4 张 A100，160GB 总显存（整节点）
```

### gpuA100x8 分区（每节点最多 8 张）

```bash
#SBATCH --partition=gpuA100x8
#SBATCH --gpus-per-node=4    # 4 张 A100，160GB
#SBATCH --gpus-per-node=8    # 8 张 A100，320GB 总显存（整节点）
```

---

## 三、推荐配置（针对本项目）

### 单 GPU 训练（显存够用时）

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
```

### 多 GPU 推理 / 显存不足时（推荐）

```bash
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4    # 160GB 总显存，足以跑 480×720
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
```

### 超大显存需求（480×720 多卡训练）

```bash
#SBATCH --partition=gpuA100x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8    # 320GB 总显存
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
```

---

## 四、关键注意事项

### `--gpus-per-node` 上限不能超过分区物理数量

- `gpuA100x4` 分区：最多 `--gpus-per-node=4`，超出会被拒绝
- `gpuA100x8` 分区：最多 `--gpus-per-node=8`

### 内存要跟上 GPU 数量

每张 A100 显存 40GB，但系统内存也要留够。经验值：

| GPU 数 | 建议 `--mem` |
|---|---|
| 1 | 64G |
| 2 | 128G |
| 4 | 128G ~ 256G |
| 8 | 256G ~ 512G |

### 多 GPU 代码层面

Delta 申请多 GPU 后，**代码需要自行管理多卡**。diffusers 的 pipeline 默认只用 GPU 0。如要用全部卡，需要：

```python
# 方法1：pipeline device_map（推理用）
pipe = CogVideoXPipeline.from_pretrained(..., device_map="balanced")

# 方法2：PyTorch DDP（训练用）
# 通过 torchrun 启动，设置 WORLD_SIZE 等环境变量
```

如果只是解决单卡 OOM 问题（如 VAE encode 480×720），申请多卡但只用 GPU 0，通过 `vae.enable_slicing()` + `vae.enable_tiling()` 就能在单卡解决，不需要 DDP。

---

## 五、快速参考：本项目常用脚本头

```bash
# 标准训练（480×720，单卡 A100 + slicing/tiling）
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# 显存吃紧的推理 / 评测（多卡）
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00

# 超长训练（接力用）
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00   # 注意：默认最长 2 天
```
