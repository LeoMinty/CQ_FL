# CQ-FL 实验一运行说明

该入口实现草稿 3.2 节的“测试准确率—通信轮次”实验，并保留原来的
`FLConfig*.py` demo 不变。

## 统一实验口径

| 数据集 | 客户端 | 轮次 | 本地 epoch | 类别 |
|---|---:|---:|---:|---:|
| RAVDESS | 2 | 50 | 2 | 8 |
| DroneRF/DRF-2 | 5 | 100 | 2 | 4 |
| MNIST Non-IID | 10 | 50 | 2 | 10 |

共同参数：batch size 32、学习率 `3e-4`、CA-4bit block size 64、默认随机种子
42。四种方法是 `fedavg_fp32`、`signsgd`、`w2_fp32_adam`、`cqfl`。

四种方法现在共用同一套复数网络和参数布局。正式入口直接复用原项目的
`BitMyConv_noMul.ComplexConv2D`、`simplified_complex_matmul`、
`Bit2Conv.ComplexMaxPool2D` 以及 `Bit2Linear.RealToComplex/TakeReal`；新包只负责
数据、实验组织、CA-4bit 状态和结果记录。

## 数据准备

RAVDESS 使用真实复数 STFT，而不是旧 demo 的 Log-Mel + delta：

```bash
python prepare_ravdess_stft.py \
  --raw-root "/path/to/RAVDESS" \
  --output ravdess_processed/ravdess_c3_stft.npz
```

DroneRF 公共原始文件是 L/H 两个实值频段，并不是原生 IQ。为让三种数据使用同一
复数网络，预处理保留两频段 FFT 的复数系数，组合成 `64x32x1x2`；DRF-2 标签为
Background/Bebop/AR/Phantom。窗口必须按物理采集段分组后再划分，以防数据泄漏。

```bash
python prepare_dronerf.py \
  --raw-root "/path/to/DroneRF" \
  --output dronerf_processed/dronerf_drf2_complex.npz
```

MNIST 由 TensorFlow 自动下载，采用经典 shard 划分，每个客户端恰好两个 shard，近似
满足“每客户端两个数字类别”。

## 云端先做冒烟测试

先用少量样本验证显存、输入形状和四种优化路径：

```bash
python run_experiment1.py --dataset mnist --method all \
  --rounds 1 --local-epochs 1 \
  --max-train-samples 3200 --max-test-samples 1000
```

冒烟测试通过后，运行正式配置：

```bash
python run_experiment1.py --dataset mnist --method all
python run_experiment1.py --dataset ravdess --method all
python run_experiment1.py --dataset dronerf --method all
```

每次运行写入独立目录：

```text
results/experiment1/<dataset>/<method>/seed_<seed>_<timestamp>/
  config.json
  metrics.csv
  summary.json
```

论文正式结果建议至少使用 3 个种子（例如 42、43、44），报告均值和标准差。实验一作图
可直接运行：

```bash
python plot_experiment1.py --dataset mnist
python plot_experiment1.py --dataset ravdess
python plot_experiment1.py --dataset dronerf
```

## 实现边界

- 2-bit 权重和梯度消息均按位打包计数；TensorFlow 计算图中的影子变量仍是 FP32，不能把
  逻辑位宽直接写成实测 GPU 显存。
- 原 BitFL 的 2-bit 单位四相位与无乘法卷积保持不变。`w2_fp32_adam` 使用标准STE，
  `cqfl` 才启用反向相位量化；两者的前向完全相同。
- 2-bit 权重量化作用于 ComplexConv kernel；复数偏置以及取实部后的 Dense/BN 权重保持
  FP32。CQ-FL 对所有复数梯度/客户端更新应用相位量化，实值分类头保持 FP32。
- CA-4bit 的一阶复数动量持久化为 2-bit 相位和 4-bit 幅度。二阶动量保持标准Adam的
  原始形状，实部和虚部各自维护独立状态，再分别使用排除零点的4-bit线性映射；每64个值
  保存一个FP16 scale。因此4-bit只改变状态存储，不把Adam改成共享分母的另一种算法。
- 客户端继续使用原BitFL的CPU FP32主权重/计算影子路径。CA-4bit取代的是持久化Adam状态，
  不是原有2-bit卷积模块。
- 联邦主循环以 FedAvg 模型增量实现本地两 epoch。CQ-FL 上传复数模型增量的 2-bit 相位编码，
  它是草稿中“上传梯度”的可执行 FedAvg 对应物；论文应称为“量化客户端更新”，除非后续把
  服务端改成逐梯度 FedOpt。
