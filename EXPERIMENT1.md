# CQ-FL 实验一运行说明

该入口实现草稿 3.2 节的“测试准确率—通信轮次”实验，并保留原来的
`FLConfig*.py` demo 不变。

## 统一实验口径

| 数据集 | 客户端 | 轮次 | 本地 epoch | 类别 |
|---|---:|---:|---:|---:|
| RAVDESS | 2 | 50 | 2 | 8 |
| DroneRF/DRF-2 | 5 | 100 | 2 | 4 |
| MNIST Non-IID | 10 | 50 | 2 | 10 |

共同参数：batch size 32、学习率 `3e-4`、CA-4bit block size 64。论文正式结果固定
使用随机种子 `42`、`123`、`2024`。四种方法是 `fedavg_fp32`、`signsgd`、
`w2_fp32_adam`、`cqfl`。

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

冒烟测试通过后，使用一个全新的正式结果目录。不要把 smoke、pilot 或不同参数的结果
写入该目录。RAVDESS、MNIST 与 DroneRF 均使用相同的三个种子：

```bash
export TF_CPP_MIN_LOG_LEVEL=3
export TF_DETERMINISTIC_OPS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for seed in 42 123 2024; do
  PYTHONHASHSEED=$seed python -u run_experiment1.py \
    --dataset ravdess --method all --seed "$seed" \
    --output-root results/experiment1_final
done

for seed in 42 123 2024; do
  PYTHONHASHSEED=$seed python -u run_experiment1.py \
    --dataset mnist --method all --seed "$seed" \
    --output-root results/experiment1_final
done

for seed in 42 123 2024; do
  PYTHONHASHSEED=$seed python -u run_experiment1.py \
    --dataset dronerf --method all --seed "$seed" \
    --output-root results/experiment1_final
done
```

每次运行写入独立目录：

```text
results/experiment1_final/<dataset>/<method>/seed_<seed>_<timestamp>/
  config.json
  metrics.csv
  summary.json
```

三个数据集都必须使用同一组种子，不能按数据集选择更有利的种子。当前实现中，RAVDESS
和 DroneRF 的 seed 同时控制数据/物理组划分、客户端分配和训练随机性；MNIST 使用固定官方
测试集，seed 控制 Non-IID shard 分配和训练随机性。论文中应明确说明这一点。

正式绘图工具默认严格要求 `42`、`123`、`2024` 各出现一次，并检查四种方法的配置和轮数
完全一致。缺少种子、重复运行、混入额外种子、配置不同或轮数不完整都会报错。运行：

```bash
python plot_experiment1.py --dataset mnist \
  --results-root results/experiment1_final --seeds 42 123 2024
python plot_experiment1.py --dataset ravdess \
  --results-root results/experiment1_final --seeds 42 123 2024
python plot_experiment1.py --dataset dronerf \
  --results-root results/experiment1_final --seeds 42 123 2024
```

每个数据集会生成一份 PDF 曲线和一份 CSV 汇总。例如：

```text
results/experiment1_final/ravdess_accuracy_vs_round.pdf
results/experiment1_final/ravdess_accuracy_summary.csv
```

如果绘图工具报告某个种子重复，应把旧的重复运行目录移出正式结果根目录，而不是让脚本
自动选择一次运行。

## 实现边界

- 2-bit 权重和梯度消息均按位打包计数；TensorFlow 计算图中的影子变量仍是 FP32，不能把
  逻辑位宽直接写成实测 GPU 显存。
- 原 BitFL 的 2-bit 单位四相位与无乘法卷积保持不变。`w2_fp32_adam` 使用标准STE，
  `cqfl` 才启用反向相位量化；两者的前向完全相同。
- 2-bit 权重量化作用于 ComplexConv kernel；复数偏置以及取实部后的 Dense/BN 权重保持
  FP32。CQ-FL 对所有复数梯度/客户端更新应用相位量化，实值分类头保持 FP32。
- CA-4bit 的一阶复数动量持久化为 2-bit 相位和 4-bit 幅度。二阶动量保持标准Adam的
  原始形状，实部和虚部各自维护独立状态，再分别使用排除零点的4-bit线性映射；每64个值
  保存一个FP32 scale。FP32 scale 是块量化元数据，用于避免Adam二阶矩在FP16范围内下溢；
  块内状态编码仍为4-bit。CA-4bit只改变状态存储，不把Adam改成共享分母的另一种算法。
- 客户端继续使用原BitFL的CPU FP32主权重/计算影子路径。CA-4bit取代的是持久化Adam状态，
  不是原有2-bit卷积模块。
- 联邦主循环以 FedAvg 模型增量实现本地两 epoch。CQ-FL 上传复数模型增量的 2-bit 相位编码，
  它是草稿中“上传梯度”的可执行 FedAvg 对应物；论文应称为“量化客户端更新”，除非后续把
  服务端改成逐梯度 FedOpt。
