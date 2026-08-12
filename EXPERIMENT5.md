# 实验 5：每参数持久化训练状态

实验 5 主结果采用以下统一口径：

1. FedAvg、BitFL、SignSGD 使用 FP32 持久权重；
2. `w2_fp32_adam`、CQ-FL 使用 2-bit packed 持久权重，每张量另存一个 FP32 scale；
3. CQ-FL 优化器使用当前 CA4 的物理 packed m/v 状态。

两种口径均只统计单客户端持久化的权重和优化器状态。梯度、激活、解码临时数组、Python
对象开销及 BN moving mean/variance 等 non-trainable state 不计入。分母严格使用 trainable
FP32 标量数，而不是包含 non-trainable state 的 `model.count_params()`。

在包含 TensorFlow 的正式实验环境运行：

```bash
python -u analyze_storage.py \
  --datasets ravdess dronerf mnist \
  --mnist-profile mnist_small \
  --block-size 64 \
  --output-dir results/experiment5_storage
```

输出包括：

- `storage_summary.csv`：实验 5 主结果；CQ-FL 应约为 `1.37 B/param`；
- `storage_all_definitions.csv`：两种口径的完整审计明细，不作为主表；
- `storage_details.json`：逐变量 shape、复数掩码及 m/v 字节分项，供审计；
- `storage_summary.md`：论文表格初稿；
- `storage_bars.png`：实验 5 主结果的 B/param 对比图。

MNIST 的 `model_profile` 尚未最终冻结。如果正式实验选择 `standard`，必须以
`--mnist-profile standard` 重新生成全部 MNIST 存储数据，不得沿用 `mnist_small` 的结果。

该统计是持久化编码格式，不是 TensorFlow 运行期实际 CPU/GPU 峰值内存。
