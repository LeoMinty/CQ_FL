# 旧代码2-bit审计与正式通信协议

## 审计结论

旧代码没有实现可直接复用的“联邦2-bit通信”。它实现的是客户端本地计算量化：

- `BitMyConv_noMul.py` 将 `ComplexConv2D` 的卷积核前向值映射到
  `{+1,+i,-1,-i}`，并可在自定义梯度中执行同一相位映射；这些仍是TensorFlow浮点张量。
- `FLConfig.py` 与 `FLConfig_ravdess.py` 在一轮本地训练后收集FP32 `W_master`，服务端直接
  求均值；没有2-bit code、位打包、消息解码或对解码值的聚合。
- 原 `README.md` 也明确说明旧demo通信量与全精度相同、暂未进行模型压缩。
- `FLConfig.py` 的随机投影/散列分支默认使用可配置的8-bit标量量化，但结果仍以浮点数组
  保存，没有2-bit位打包，也不是四相位或多epoch FedAvg模型增量，不能作为该论文方案使用。

因此，正式代码继续复用旧的 `BitMyConv_noMul.ComplexConv2D` 作为2-bit本地计算核心，并由
新增的 `Bit2Communication.py` 补齐旧项目缺失的联邦通信边界。CA-4bit仍只替换客户端Adam
持久状态，不替换旧卷积层。

## CQ-FL上行定义

客户端完成本地训练后计算相对于本轮广播基点的模型增量。所有可训练张量均经过真实的
2-bit打包、解包和重构，服务端聚合重构值：

1. 复数卷积kernel/bias：每个复数元素按相位映射为0/1/2/3，分别表示
   `+1/+i/-1/-i`，每个张量附带一个FP32平均幅值。
2. 实数Dense/BN：把实值视作虚部为0的退化复数，正数、零、负数分别编码为0、1、2；
   每个张量附带一个FP32平均绝对值。虽然只使用三个码，仍固定占2 bit。
3. 每四个2-bit码实际装入一个 `uint8`，逻辑数值负载按真实buffer字节数加FP32 scale计算；
   不含shape、count、scheme、客户端ID等消息元数据和网络协议头。
4. BN moving mean/variance等非可训练状态保持FP32，并计入 `uplink_bytes`。
5. 服务端执行
   `base + sum(n_k / sum(n) * decode(message_k))`，不会聚合未量化的隐藏副本。

计数单位必须区分：复数张量的一个码对应一对FP32实/虚分量，主项相对8字节复数是32×；
实数张量的一个码对应一个FP32标量，主项是16×。总压缩比还要加上逐张量FP32 scale和FP32
非可训练状态，不能简单把“参数总数×2 bit”作为所有张量的字节数。

这一定义压缩的是一轮本地训练后的“客户端模型增量”，是FedAvg中“上传梯度”的可执行
对应物。论文中宜称“2-bit量化客户端更新”。

## 方法边界与论文表述

旧代码和当前草稿只给出了复数四相位量化，没有给出TakeReal之后Dense/BN实数更新的2-bit
规则。上述实轴codec是为了实现“全部可训练参数2-bit上行”而新增的域感知扩展：它的有效
方向等价于“正/零/负 + 每张量平均幅值”，不是复相位梯度本身，第四个码保留未用。因此论文
不能再笼统写成“所有参数都采用四相位量化”，应分别定义复数相位codec与实数实轴codec，
并把每客户端全局scale记法改成逐张量的 `s_{i,l}`。若共有 `L` 个可训练张量、`D_c` 个
复数元素和 `D_r` 个实数元素，则不含对齐和非训练状态的主开销为
`2D_c + 2D_r + 32L` bit，而不是 `2D + 16` bit。

通信scale使用FP32，从而避免此前CA-4bit中已出现过的FP16小量下溢；编码器也会拒绝
NaN/Inf和超出FP32 scale范围的复数更新。复数四相位协议仍没有独立零码：非零张量内部恰好
为零的复数元素会按 `atan2(0,0)` 进入 `+R` 方向并被重构为共享scale。该行为由测试固定记录。
正式长轮次实验前必须先做5轮pilot；若出现收敛停滞，应在论文算法层面增加误差反馈或重新
定义量化规则，不能只修改绘图字节数。

## 与CA-4bit scale的区别

通信消息中的FP32 scale每个张量一个，只用于恢复该轮2-bit上行增量；CA-4bit优化器中的
FP32 scale每64个状态值一个，用于避免Adam二阶矩下溢。二者作用、粒度和生命周期均不同。

## 结果兼容性

旧CQ-FL结果只有复数张量被压缩，Dense/BN更新仍以FP32参与聚合，因此不能用于新版实验1
或“全可训练参数2-bit上行”的实验4。FedAvg、SignSGD、`w2_fp32_adam`算法未因本次改动
改变；CQ-FL三个种子必须重新运行。新版 `metrics.csv` 会额外记录：

- `uplink_trainable_bytes`
- `uplink_complex_2bit_bytes`
- `uplink_real_2bit_bytes`
- `uplink_non_trainable_bytes`

`plot_experiment4.py` 会验证这些分项并拒绝旧CQ-FL文件。

按当前模型形状和默认客户端数，新版CQ-FL每轮上行逻辑负载的静态期望值如下，可用于核对
服务器首轮输出文件：

| 数据集/模型 | FedAvg每轮 | CQ-FL每轮 | 理论比值 |
|---|---:|---:|---:|
| RAVDESS standard，2客户端 | 50,985,024 B | 3,178,468 B | 16.04× |
| DroneRF standard，5客户端 | 12,108,880 B | 736,565 B | 16.44× |
| MNIST `mnist_small`，10客户端 | 2,143,120 B | 135,070 B | 15.87× |

该比值是上行逻辑负载比，不是实测网络吞吐或GPU显存比。若模型结构或客户端数改变，数值也
会相应改变。

## 验证

上传新代码后先运行：

```bash
python -m unittest -v test_bit2_communication.py test_federated_wiring.py
python -m compileall -q Bit2Communication.py cqfl run_experiment1.py \
  plot_experiment1.py plot_experiment4.py validate_bit2_run.py
```

测试覆盖2-bit边界相位、任意长度打包/解包、零张量、实值Dense消息、真实buffer字节数、
服务端对解码增量的加权聚合，以及“输出维恰好为2的实值Dense不得被误判为复数张量”。
没有安装TensorFlow的预处理机器会自动跳过第二个测试文件；实验服务器不能跳过它。

随后在服务器进行一轮极小接线测试（只用于验证，不进入正式结果目录）：

```bash
python -u run_experiment1.py \
  --dataset ravdess --method cqfl --seed 42 \
  --rounds 1 --local-epochs 1 --batch-size 32 \
  --max-train-samples 64 --max-test-samples 32 \
  --output-root results/bit2_wiring_smoke
```

命令结束会打印实际run目录，把该目录原样传给验证器，例如：

```bash
python validate_bit2_run.py \
  results/bit2_wiring_smoke/ravdess/cqfl/seed_42_YYYYMMDD_HHMMSS
```

验证器应输出 `uplink/round: 3,178,468 B`，并确认复数2-bit、实数2-bit和FP32非可训练
状态三项均存在。该测试通过后先运行同配置的5轮完整样本CQ-FL pilot，再决定是否启动三个
种子的正式重跑。
