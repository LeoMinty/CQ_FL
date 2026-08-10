import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np

# 训练参数
Eps = 10.
epochs = 50
batch_size = 1024

def stable_dp_axis_selection(angles, epsilon):
    """
    稳定的差分隐私轴选择
    当ε很大时，行为应该接近原始确定性算法
    """
    # 四个轴的角度中心
    axis_centers = tf.constant([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=tf.float32)

    # 将角度归一化到[0, 2π)
    angles_norm = tf.math.floormod(angles + 2 * np.pi, 2 * np.pi)

    # 扩展维度以便广播计算
    angles_expanded = tf.expand_dims(angles_norm, axis=-1)  # [..., 1]
    centers_expanded = tf.expand_dims(axis_centers, axis=0)  # [1, 4]

    # 计算角度差并考虑圆周性
    diff = tf.abs(angles_expanded - centers_expanded)
    distances = tf.minimum(diff, 2 * np.pi - diff)  # [..., 4]

    # 角度的敏感值应为pi
    sensitivity = np.pi

    # 使用更稳定的计算方式
    scores = -epsilon * distances / (2 * sensitivity)

    # 减去最大值以提高数值稳定性
    max_scores = tf.reduce_max(scores, axis=-1, keepdims=True)
    stable_scores = scores - max_scores
    weights = tf.exp(stable_scores)

    # 归一化得到概率
    weights_sum = tf.reduce_sum(weights, axis=-1, keepdims=True)
    probs = weights / tf.maximum(weights_sum, 1e-6)

    # 为每个角度采样选择的轴
    original_shape = tf.shape(angles)
    flat_probs = tf.reshape(probs, [-1, 4])
    choices = tf.random.categorical(tf.math.log(flat_probs), 1)
    choices = tf.reshape(choices, original_shape)

    # 根据选择创建四个区域的掩码
    real_pos = tf.equal(choices, 0)
    imag_pos = tf.equal(choices, 1)
    real_neg = tf.equal(choices, 2)
    imag_neg = tf.equal(choices, 3)

    return real_pos, real_neg, imag_pos, imag_neg

def differential_private_axis_selection(angles, epsilon):
    """
    差分隐私的轴选择：从四个轴中随机选择一个，概率与角度距离相关
    """
    # 计算四个轴的角度中心 [0°, 90°, 180°, 270°]
    axis_centers = tf.constant([0.0, np.pi / 2, np.pi, 3 * np.pi / 2], dtype=tf.float32)

    # 将角度归一化到[0, 2π)
    angles_norm = tf.math.floormod(angles + 2 * np.pi, 2 * np.pi)

    # 向量化计算角度距离
    # 扩展维度以便广播计算
    angles_expanded = tf.expand_dims(angles_norm, axis=-1)  # [..., 1]
    centers_expanded = tf.expand_dims(axis_centers, axis=0)  # [1, 4]

    # 计算角度差并考虑圆周性
    diff = tf.abs(angles_expanded - centers_expanded)
    distances = tf.minimum(diff, 2 * np.pi - diff)  # [..., 4]

    # 使用指数机制：距离越小，概率越大
    # 敏感度 = π (最大角度距离)
    sensitivity = np.pi
    weights = tf.exp(-epsilon * distances / (2 * sensitivity))

    # 归一化得到概率
    weights_sum = tf.reduce_sum(weights, axis=-1, keepdims=True)
    probs = weights / tf.maximum(weights_sum, 1e-8)

    # 为每个角度采样选择的轴
    log_probs = tf.math.log(probs)

    # 处理任意形状的输入
    original_shape = tf.shape(angles)
    flat_log_probs = tf.reshape(log_probs, [-1, 4])
    choices = tf.random.categorical(flat_log_probs, 1)
    choices = tf.reshape(choices, original_shape)

    # 根据选择创建四个区域的掩码
    real_pos = tf.equal(choices, 0)  # 选择第一个轴 (0°方向)
    imag_pos = tf.equal(choices, 1)  # 选择第二个轴 (90°方向)
    real_neg = tf.equal(choices, 2)  # 选择第三个轴 (180°方向)
    imag_neg = tf.equal(choices, 3)  # 选择第四个轴 (270°方向)

    return real_pos, real_neg, imag_pos, imag_neg

# 1. 定义 PhaseQuant 量化函数
def phase_quant(complex_weights, epsilon=1.):
    """
    将复数权重量化为四个方向并应用缩放因子
    输入: 复数张量，形状为 (..., 2)，最后一维是 [实部, 虚部]
    输出: 量化后的复数张量，形状与输入相同
    """
    real = complex_weights[..., 0]
    imag = complex_weights[..., 1]

    # 计算相位角（弧度制）
    angles = tf.math.atan2(imag, real)

    # 创建区域掩码
    #real_pos = tf.logical_and(angles >= -np.pi / 4, angles < np.pi / 4)
    #real_neg = tf.logical_or(angles >= 3 * np.pi / 4, angles < -3 * np.pi / 4)
    #imag_pos = tf.logical_and(angles >= np.pi / 4, angles < 3 * np.pi / 4)
    #imag_neg = tf.logical_and(angles >= -3 * np.pi / 4, angles < -np.pi / 4)

    # 使用差分隐私方法选择轴（替换原来的四行代码）
    real_pos, real_neg, imag_pos, imag_neg = stable_dp_axis_selection(angles, epsilon)

    # 计算缩放因子
    real_mask = tf.logical_or(real_pos, real_neg)
    imag_mask = tf.logical_or(imag_pos, imag_neg)

    # 防止除零错误
    eps = 1e-6

    # 实部缩放因子：实轴区域实部绝对值的均值
    real_scale = tf.reduce_mean(tf.abs(tf.where(real_mask, real, 0.0)))
    real_scale = 1.0 / tf.maximum(real_scale, eps)

    # 虚部缩放因子：虚轴区域虚部绝对值的均值
    imag_scale = tf.reduce_mean(tf.abs(tf.where(imag_mask, imag, 0.0)))
    imag_scale = 1.0 / tf.maximum(imag_scale, eps)

    # 初始化量化值
    quant_real = tf.zeros_like(real)
    quant_imag = tf.zeros_like(imag)

    # 分配量化值
    quant_real = tf.where(real_pos, 1.0, quant_real)
    quant_real = tf.where(real_neg, -1.0, quant_real)
    quant_imag = tf.where(imag_pos, 1.0, quant_imag)
    quant_imag = tf.where(imag_neg, -1.0, quant_imag)

    # 应用缩放因子
    quant_real = quant_real / real_scale
    quant_imag = quant_imag / imag_scale

    return tf.stack([quant_real, quant_imag], axis=-1)

def differential_private_phase_quant_vectorized(complex_weights, epsilon):
    """
    向量化的差分隐私随机相位量化（更高效）
    """
    real = complex_weights[..., 0]
    imag = complex_weights[..., 1]

    # 防止除零错误
    epsilon_safe = tf.maximum(epsilon, 1e-5)
    tanh_eps_half = tf.math.tanh(epsilon_safe / 2)

    # 计算四个方向的概率
    p_real_pos = 0.5 + 0.5 * tanh_eps_half * real
    p_real_neg = 1 - p_real_pos
    p_imag_pos = 0.5 + 0.5 * tanh_eps_half * imag
    p_imag_neg = 1 - p_imag_pos

    # 归一化概率
    p_real_pos_norm = p_real_pos / 2
    p_real_neg_norm = p_real_neg / 2
    p_imag_pos_norm = p_imag_pos / 2
    p_imag_neg_norm = p_imag_neg / 2

    # 构建概率张量
    probs = tf.stack([p_real_pos_norm, p_real_neg_norm, p_imag_pos_norm, p_imag_neg_norm], axis=-1)

    # 为每个元素采样方向选择
    log_probs = tf.math.log(probs)
    choices = tf.random.categorical(tf.reshape(log_probs, [-1, 4]), 1)
    choices = tf.reshape(choices, tf.shape(real))

    # 根据选择设置量化值
    quant_real = tf.where(choices == 0, 1.0,
                          tf.where(choices == 1, -1.0, 0.0))
    quant_imag = tf.where(choices == 2, 1.0,
                          tf.where(choices == 3, -1.0, 0.0))

    return tf.stack([quant_real, quant_imag], axis=-1)

# 2. 定义复数线性层
class ComplexLinear(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, quant_weight_decay=1e-1,  # 量化正则化强度
                 **kwargs):
        super(ComplexLinear, self).__init__(**kwargs)
        self.units = units
        self.use_bias = use_bias
        self.quant_weight_decay = quant_weight_decay

    def build(self, input_shape):
        # 输入形状: (batch_size, input_dim, 2) - 最后一维是实部和虚部
        input_dim = input_shape[-2]

        # 初始化全精度复数权重
        self.w_fp = self.add_weight(
            name="w_fp",
            shape=(input_dim, self.units, 2),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units, 2),
                initializer="zeros",
                trainable=True
            )

        super(ComplexLinear, self).build(input_shape)

    def phase_quant_with_ste(self, weights):
        """
        使用直通估计器(STE)的权重量化
        前向传播：量化到离散值 {1, i, -1, -i}
        反向传播：梯度直接传递（直通）
        """
        # 先对权重进行裁剪（-1,1），保留裁剪操作的梯度
        weights = tf.clip_by_value(weights, -1.0, 1.0)
        # 前向传播：量化权重
        quantized = phase_quant(weights, epsilon=Eps)

        # 反向传播：使用直通估计器（梯度直接传递）
        return weights + tf.stop_gradient(quantized - weights)

    def add_quantization_regularization(self):
        """
        添加量化正则化损失，促使权重逼近离散值
        """
        if self.quant_weight_decay > 0:
            # 计算量化权重
            quantized = phase_quant(self.w_fp, epsilon=Eps)

            # 计算全精度权重与量化权重之间的L2距离
            quantization_loss = tf.reduce_mean(tf.square(self.w_fp - quantized))
            kernel_loss = tf.reduce_mean(tf.abs(self.w_fp))
            # 添加正则化损失
            self.add_loss(self.quant_weight_decay * quantization_loss)
            self.add_loss(self.quant_weight_decay * kernel_loss)

    def call(self, inputs, training=None):
        # 前向传播: 使用量化权重
        w_quant = self.phase_quant_with_ste(self.w_fp)
        # 拆分输入和权重的实部和虚部
        real_in = inputs[..., 0]  # 实部 (batch_size, input_dim)
        imag_in = inputs[..., 1]  # 虚部 (batch_size, input_dim)

        w_real = w_quant[..., 0]  # 实部 (input_dim, units)
        w_imag = w_quant[..., 1]  # 虚部 (input_dim, units)

        # 复数矩阵乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # 但由于权重已被量化为 {1, i, -1, -i}，我们可以简化计算

        # 计算实部: ac - bd
        # 计算虚部: ad + bc

        # 由于权重是离散值，我们可以直接计算
        real_out = tf.matmul(real_in, w_real) - tf.matmul(imag_in, w_imag)
        imag_out = tf.matmul(real_in, w_imag) + tf.matmul(imag_in, w_real)

        # 添加偏置
        if self.use_bias:
            real_out += self.bias[..., 0]
            imag_out += self.bias[..., 1]

        self.add_quantization_regularization()

        return tf.stack([real_out, imag_out], axis=-1)

    def get_config(self):
        config = super(ComplexLinear, self).get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
            'quant_weight_decay': self.quant_weight_decay
        })
        return config

# 3. 定义将实数输入转换为复数表示的层
class RealToComplex(tf.keras.layers.Layer):
    def call(self, inputs):
        # 将实数输入转换为复数表示: 实部为输入值，虚部为0
        return tf.stack([inputs, tf.zeros_like(inputs)], axis=-1)


# 4. 定义只取复数实部的层
class TakeReal(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[..., 0]


# 5. 构建模型
def create_ifairy_model(input_dim=784, num_classes=10):
    model = tf.keras.Sequential([
        tf.keras.layers.Reshape((input_dim,), input_shape=(28, 28)),
        RealToComplex(),
        ComplexLinear(256),
        tf.keras.layers.Lambda(lambda x: tf.stack([
            tf.nn.relu(x[..., 0]),  # 对实部应用ReLU
            tf.nn.relu(x[..., 1])  # 对虚部应用ReLU
        ], axis=-1)),
        ComplexLinear(128),
        tf.keras.layers.Lambda(lambda x: tf.stack([
            tf.nn.relu(x[..., 0]),  # 对实部应用ReLU
            tf.nn.relu(x[..., 1])  # 对虚部应用ReLU
        ], axis=-1)),
        ComplexLinear(64),
        tf.keras.layers.Lambda(lambda x: tf.stack([
            tf.nn.relu(x[..., 0]),  # 对实部应用ReLU
            tf.nn.relu(x[..., 1])  # 对虚部应用ReLU
        ], axis=-1)),
        ComplexLinear(num_classes),
        TakeReal(),  # 只取实部用于分类
        tf.keras.layers.Softmax()
    ])
    return model

# 6. 自定义训练步骤，使用直通估计器(STE)
@tf.function
def train_step(model, x, y, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        # 前向传播使用量化权重
        y_pred = model(x, training=True)
        loss = loss_fn(y, y_pred)

    # 获取所有可训练变量
    trainable_vars = model.trainable_variables

    # 计算梯度
    gradients = tape.gradient(loss, trainable_vars)

    # 使用优化器更新权重
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss


# 7. 主训练循环
def train_ifairy_model():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # 创建模型
    model = create_ifairy_model()

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=1e-3, weight_decay=1e-4), loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    # 按照 val_acc 的值来保存模型的参数，val_acc 有提升才保存新的参数
    callback = tf.keras.callbacks.ModelCheckpoint('checkpoints/Bit2Linear/weights-2025-10-02.h5',
                                                  monitor='val_acc',
                                                  save_best_only=True, )
    # 模型训练若干个epochs
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15, callbacks=[callback, ])
    # 模型保存本地
    # model.save("./saver/MyTfModelForMnist.h5")
    model.load_weights('checkpoints/Bit2Linear/weights-2025-10-02.h5')
    # 模型在测试集上的评估
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("测试集准确率:", score)  # 输出 [损失率，准确率]
    return model

# 8. 运行训练
if __name__ == "__main__":
    model = train_ifairy_model()