# CQ-FL — 复数神经网络量化与联邦学习实验框架

## 概述

本项目探索**复数域神经网络的极端低比特量化**及其在**联邦学习**场景下的应用。核心思想是将网络权重和梯度同时量化到复平面上的 4 个方向 `{1, i, -1, -i}`（即 2-bit 相位量化），从而大幅降低 GPU 显存占用和通信带宽需求。

主要工作包括：

1. **复数网络基础组件**：实现了复数卷积 (`ComplexConv2D`)、复数批归一化 (`ComplexBatchNormalization`)、复数线性层 (`ComplexLinear`)、复数最大池化 (`ComplexMaxPool2D`) 等算子
2. **2-bit 相位量化**：前向时通过 `phase_quant` 将 fp32 权重按相位角映射到 `{1, i, -1, -i}`，反向时梯度同样量化（LPT-FL 风格）
3. **无乘法卷积**：量化后的复数矩阵乘法 `simplified_complex_matmul` 利用掩码 + matmul 替代逐元素乘法，硬件友好
4. **CPU-Offload 训练**：fp32 主权重 + Adam 状态常驻 CPU，GPU 仅保留量化后的 2-bit 权重和临时激活值
5. **联邦学习 (FedAvg)**：多客户端本地训练 → 权重聚合，支持梯度量化通信

---

## 项目结构

```
CQ-FL/
├── README.md                     # 本文件
├── requirements.txt              # 依赖
│
├── BitMyConv_noMul.py            # ★ 核心模块：无乘法 ComplexConv2D + ComplexBN
├── Bit2Conv.py                   # 复数池化 (ComplexMaxPool2D) + 基础复数卷积参考实现
├── Bit2Linear.py                 # 复数线性层 (ComplexLinear) + RealToComplex/TakeReal
├── BitMyConv.py                  # 复数卷积模型（早期版本，含 STE 量化 + 缩放因子）
│
├── FLConfig.py                   # ★ 联邦学习主入口 (MNIST) — 含 3 种模式
├── FLConfig_ravdess.py           # ★ 联邦学习主入口 (RAVDESS 语音情感识别)
├── test_ravdess_models.py        # RAVDESS 模型对比测试（实数 vs 复数 vs 量化）
├── prepare_ravdess.py            # RAVDESS 数据预处理脚本
├── DataProcessing.py             # 音频 STFT 特征提取工具
│
├── BitMyConv_noMul_offload.py    # CPU-Offload 单机训练 (MNIST)
├── NonBitConv.py                 # 实值自定义卷积层 (EfficientCustomConv2D，基线对比)
├── NonBitLinear.py               # 实值 MLP 基线 (Fashion MNIST)
│
├── test.py                       # 差分隐私 Count Sketch 实验
├── test2.py                      # 差分隐私对比实验 (策略A vs 策略B)
│
├── ravdess_processed/            # 预处理后的 RAVDESS 数据
│   └── ravdess_c3.npz           # c3=情绪分类(8类)
├── results/                      # 训练结果输出目录
└── temp/                         # 临时文件
```

正式论文实验一另外使用：

```text
run_experiment1.py               # 三数据集、四方案统一入口
cqfl/models.py                   # 组装原BitFL层，不重复实现ComplexConv
cqfl/federated.py                # CPU主权重、客户端训练与FedAvg
cqfl/ca4bit.py                   # CPMQ一阶矩 + 逐分量4-bit二阶矩
cqfl/quantization.py             # 2-bit通信打包/解包
cqfl/data.py                     # 三数据集与客户端划分
prepare_ravdess_stft.py          # 正式RAVDESS复数STFT
prepare_dronerf.py               # DroneRF复数FFT及物理段编号
```

其中正式实验仍直接调用 `BitMyConv_noMul.py`、`Bit2Conv.py` 和
`Bit2Linear.py` 的原有核心模块；`cqfl/` 是实验组织与4-bit优化器扩展层。

---

## 核心技术原理

### 1. 2-bit 相位量化 (Phase Quantization)

将任意复数权重按相位角映射到 4 个基向量：

```
角度范围             量化结果
[  0°,  45°) ∪ [315°, 360°)  →  1  (实轴正)
[ 45°, 135°)                  →  i  (虚轴正)
[135°, 225°)                  → -1  (实轴负)
[225°, 315°)                  → -i  (虚轴负)
```

- **前向**：权重 phase_quant 为 `{1, i, -1, -i}`，卷积/矩阵乘可通过位操作或无乘法实现
- **反向**：`custom_gradient` 将梯度也量化（STE 直通估计的扩展），梯度变成 2-bit 后再回传 CPU

### 2. 无乘法复数矩阵乘

量化后的复数 matmul `(a+bi) × {1,-1,i,-i}` 退化为加减法：

```
B = 1:   C =  a + bi    (不变)
B = -1:  C = -a - bi    (取负)
B = i:   C = -b + ai    (交换+负)
B = -i:  C =  b - ai    (交换+负)
```

实现上使用 4 个掩码 + `tf.matmul` 批次，避免显式的乘法运算和 tile 展开导致的 OOM。

### 3. CPU-Offload 架构（论文中已舍弃此idea）

```
┌──────────── CPU ────────────┐     ┌──────────── GPU ────────────┐
│  fp32 W_master               │────→│  2-bit 量化权重（无乘法卷积） │
│  Adam 优化器状态             │     │  激活值（临时）              │
│  ← 2-bit 梯度（反向）        │←────│  custom_gradient 量化梯度    │
│  Adam update                 │     │                              │
└──────────────────────────────┘     └──────────────────────────────┘
```

每步训练：CPU→GPU(权重同步) → GPU前向/反向 → GPU→CPU(量化梯度) → CPU Adam更新

### 4. 联邦学习 (FedAvg)

- 每轮：各客户端从全局 `W_master` 复制权重 → 本地训练 2 epoch → 上传 `W_master`
- 聚合：对所有客户端和 BN 统计量取均值（CPU 侧完成）
- 通信量：与全精度模型相同（暂未做模型压缩），但训练时的梯度已是 2-bit

---

## 使用方式

> 论文草稿中的三数据集、四方案“实验一”已迁移到独立入口。云端运行和数据准备命令见
> [EXPERIMENT1.md](EXPERIMENT1.md)。原有 `FLConfig.py` 与 `FLConfig_ravdess.py`
> 仅作为早期 demo 保留，不应再用于论文正式结果。

### 环境准备

```bash
pip install -r requirements.txt
```

### 数据预处理 (RAVDESS)

```bash
# 修改 prepare_ravdess.py 中的 DATA_DIR 指向你的 RAVDESS 数据集路径
# 预期目录结构: DATA_DIR/Actor_01/*.wav ... DATA_DIR/Actor_24/*.wav
python prepare_ravdess.py
# 输出: ./ravdess_processed/ravdess_c3.npz (情绪分类, 8类)
```

数据集中已有预处理好的 `ravdess_processed/ravdess_c3.npz`。

### 运行联邦学习

**RAVDESS 语音情感识别 (2 客户端 FedAvg + CPU-Offload)**：

```bash
python FLConfig_ravdess.py
```

关键配置参数（在文件顶部修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `USE_OFFLOAD` | True | CPU-Offload 模式 |
| `NUM_CLIENTS` | 2 | 联邦学习客户端数 |
| `NUM_ROUNDS` | 50 | 联邦训练轮数 |
| `BATCH_SIZE` | 32 | 批次大小 |
| `EPOCHS_PER_CLIENT` | 2 | 每轮每客户端本地训练 epoch |
| `LR` | 3e-4 | 学习率 |

**MNIST 联邦学习 (支持 3 种模式切换)**：

```bash
python FLConfig.py
```

通过修改文件顶部常量切换模式：
- `USE_OFFLOAD=True` → CPU-Offload + 梯度量化 FedAvg
- `USE_HASHING=True` → 散列化梯度压缩（实验性）
- 两者均为 False → 普通 FedAvg（基线）

### 模型对比测试 (RAVDESS)

```bash
python test_ravdess_models.py
```

自动对比 3 种模型在 RAVDESS 情绪识别上的准确率：
1. 实数 CNN（Re/Im 拼接为 2 通道）
2. ComplexConv（fp32，无量化）
3. ComplexConv（2-bit 量化）

### 单机 CPU-Offload 训练

```bash
python BitMyConv_noMul_offload.py
```

MNIST 上的单机 Offload 验证（无联邦）。

---

## 关键类与函数说明

### `ComplexConv2D` (BitMyConv_noMul.py)

核心复数卷积层，支持量化/非量化两种模式：

```python
ComplexConv2D(
    filters=16,          # 输出通道数
    kernel_size=3,       # 卷积核尺寸
    padding='same',      # 填充方式
    activation=tf.nn.relu,  # 激活函数
    use_quant=True,          # True=2-bit量化, False=fp32标准卷积
    quantize_backward=True   # True=2-bit梯度, False=标准STE
)
```

- `use_quant=True, quantize_backward=True`：原BitFL/CQ-FL，前向无乘法2-bit卷积，反向2-bit相位梯度
- `use_quant=True, quantize_backward=False`：2-bit W + FP32 Adam消融，前向相同但反向使用标准STE
- `use_quant=False`：用标准 `tf.nn.conv2d` 实现复数卷积，正常梯度流

### `OffloadTrainer` (FLConfig.py / FLConfig_ravdess.py)

CPU-Offload 训练器：

- `train_step(x_batch, y_batch)`：单步训练（CPU→GPU→前向→反向→CPU更新）
- `evaluate(x_test, y_test)`：分批评估
- `_sync_cpu_to_gpu()`：CPU 权重 → GPU 模型变量
- `_phase_quant(tensor)`：梯度量化（仅用于 Dense/BN 层补齐）

### `simplified_complex_matmul` (BitMyConv_noMul.py)

量化后的无乘法复数矩阵乘，利用 4 个掩码将 `{1,-1,i,-i}` 权重下的复数乘降为加减：

```python
C = simplified_complex_matmul(A, B)
# A: (m, n, 2), B: (n, k, 2) 且 B ∈ {1,-1,i,-i}
# C: (m, k, 2)
```

### `ComplexBatchNormalization` (BitMyConv_noMul.py)

分别对实部和虚部应用实值 BN。

### `RealToComplex` / `TakeReal` (Bit2Linear.py)

- `RealToComplex`：实值输入 → 复数 `(x, 0)`
- `TakeReal`：复数 `(a, b)` → 实部 `a`

---

## 实验记录

### RAVDESS 情绪识别 (8 类)

| 模型 | 说明 | 预期表现 |
|------|------|----------|
| 随机基线 | 1/8 | ~0.125 |
| 实数 CNN | Re+Im 双通道 | 基线 |
| ComplexConv (fp32) | 无量化复数卷积 | 与实线可比 |
| ComplexConv (量化) | 2-bit 量化 | 理论略低于 fp32 |
| FL + Offload | FedAvg + CPU-Offload | 分布式训练 |

---

## 依赖

- Python >= 3.8
- TensorFlow >= 2.10 (GPU 推荐)
- NumPy
- librosa (仅数据预处理)
- matplotlib (可选，仅 test.py/test2.py 绘图用)

详见 `requirements.txt`。

---

## 许可证

学术研究用途。
