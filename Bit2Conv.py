import tensorflow as tf
import numpy as np

from Bit2Linear import ComplexLinear

# 1. 定义 PhaseQuant 量化函数（与之前相同）
def phase_quant(complex_weights):
    """
    将复数权重量化为 {1, i, -1, -i} 四个值之一
    输入: 复数张量，形状为 (..., 2)，最后一维是 [实部, 虚部]
    输出: 量化后的复数张量，形状与输入相同
    """
    real = complex_weights[..., 0]
    imag = complex_weights[..., 1]

    # 计算相位角
    angles = tf.math.atan2(imag, real)

    # 将角度映射到 [0, 2π) 范围
    angles = tf.math.floormod(angles + 2 * np.pi, 2 * np.pi)

    # 根据相位角确定量化值
    quant_real = tf.where(
        (angles < np.pi / 4) | (angles >= 7 * np.pi / 4), 1.0,
        tf.where(
            (angles >= 3 * np.pi / 4) & (angles < 5 * np.pi / 4), -1.0,
            0.0
        )
    )

    quant_imag = tf.where(
        (angles >= np.pi / 4) & (angles < 3 * np.pi / 4), 1.0,
        tf.where(
            (angles >= 5 * np.pi / 4) & (angles < 7 * np.pi / 4), -1.0,
            0.0
        )
    )

    return tf.stack([quant_real, quant_imag], axis=-1)


# 2. 定义复数卷积层
class ComplexConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', use_bias=True, activation=None, **kwargs):
        super(ComplexConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation) if activation else None

    def build(self, input_shape):
        # 输入形状: (batch_size, height, width, channels, 2) - 最后一维是实部和虚部
        input_channels = input_shape[-2]

        # 初始化全精度复数卷积核
        self.kernel_fp = self.add_weight(
            name="kernel_fp",
            shape=(self.kernel_size[0], self.kernel_size[1], input_channels, self.filters, 2),
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters, 2),
                initializer="zeros",
                trainable=True
            )

        super(ComplexConv2D, self).build(input_shape)

    def phase_quant_with_ste(self, weights):
        """STE 直通估计器：前向量化，反向直通"""
        weights = tf.clip_by_value(weights, -1.0, 1.0)
        quantized = phase_quant(weights)
        return weights + tf.stop_gradient(quantized - weights)

    def call(self, inputs, training=None):
        # 前向传播: 使用量化权重（STE 直通估计器）
        kernel_quant = self.phase_quant_with_ste(self.kernel_fp)

        # 拆分输入和权重的实部和虚部
        real_in = inputs[..., 0]  # 实部 (batch_size, height, width, channels)
        imag_in = inputs[..., 1]  # 虚部 (batch_size, height, width, channels)

        kernel_real = kernel_quant[..., 0]  # 实部 (kernel_h, kernel_w, in_channels, out_channels)
        kernel_imag = kernel_quant[..., 1]  # 虚部 (kernel_h, kernel_w, in_channels, out_channels)

        # 复数卷积: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # 计算实部: conv(real_in, kernel_real) - conv(imag_in, kernel_imag)
        # 计算虚部: conv(real_in, kernel_imag) + conv(imag_in, kernel_real)

        real_real = tf.nn.conv2d(
            real_in, kernel_real, strides=self.strides, padding=self.padding
        )
        imag_imag = tf.nn.conv2d(
            imag_in, kernel_imag, strides=self.strides, padding=self.padding
        )
        real_imag = tf.nn.conv2d(
            real_in, kernel_imag, strides=self.strides, padding=self.padding
        )
        imag_real = tf.nn.conv2d(
            imag_in, kernel_real, strides=self.strides, padding=self.padding
        )

        real_out = real_real - imag_imag
        imag_out = real_imag + imag_real

        # 添加偏置
        if self.use_bias:
            real_out += self.bias[..., 0]
            imag_out += self.bias[..., 1]

        result = tf.stack([real_out, imag_out], axis=-1)

        # 应用激活函数
        if self.activation is not None:
            result = tf.stack([
                self.activation(result[..., 0]),
                self.activation(result[..., 1])
            ], axis=-1)

        return result


# 3. 定义复数最大池化层
class ComplexMaxPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid', **kwargs):
        super(ComplexMaxPool2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides if strides else pool_size
        self.padding = padding.upper()

    def call(self, inputs):
        # 分别对实部和虚部进行最大池化
        real = inputs[..., 0]
        imag = inputs[..., 1]

        real_pooled = tf.nn.max_pool2d(
            real, ksize=self.pool_size, strides=self.strides, padding=self.padding
        )
        imag_pooled = tf.nn.max_pool2d(
            imag, ksize=self.pool_size, strides=self.strides, padding=self.padding
        )

        return tf.stack([real_pooled, imag_pooled], axis=-1)

    def get_config(self):
        config = super(ComplexMaxPool2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config


# 4. 定义将实数输入转换为复数表示的层
class RealToComplex(tf.keras.layers.Layer):
    def call(self, inputs):
        # 将实数输入转换为复数表示: 实部为输入值，虚部为0
        return tf.stack([inputs, tf.zeros_like(inputs)], axis=-1)


# 5. 定义只取复数实部的层
class TakeReal(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[..., 0]


# 6. 定义复数激活函数层
class ComplexReLU(tf.keras.layers.Layer):
    def call(self, inputs):
        # 分别对实部和虚部应用ReLU
        real = tf.nn.relu(inputs[..., 0])
        imag = tf.nn.relu(inputs[..., 1])
        return tf.stack([real, imag], axis=-1)


# 7. 定义复数展平层
class ComplexFlatten(tf.keras.layers.Layer):
    def call(self, inputs):
        # 分别展平实部和虚部，然后堆叠
        real_flat = tf.keras.layers.Flatten()(inputs[..., 0])
        imag_flat = tf.keras.layers.Flatten()(inputs[..., 1])
        return tf.stack([real_flat, imag_flat], axis=-1)


# 8. 构建复数卷积神经网络模型
def create_complex_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    model = tf.keras.Sequential([
        # 输入层
        tf.keras.layers.InputLayer(input_shape=input_shape),

        # 转换为复数表示
        RealToComplex(),

        # 第一个复数卷积块
        ComplexConv2D(filters=32, kernel_size=(3, 3), padding='same'),
        ComplexReLU(),
        ComplexMaxPool2D(pool_size=(1, 1)),

        # 第二个复数卷积块
        ComplexConv2D(filters=64, kernel_size=(3, 3), padding='same'),
        ComplexReLU(),
        ComplexMaxPool2D(pool_size=(1, 1)),

        # 第三个复数卷积块
        ComplexConv2D(filters=128, kernel_size=(3, 3), padding='same'),
        ComplexReLU(),
        ComplexMaxPool2D(pool_size=(1, 1)),

        # 展平
        ComplexFlatten(),

        # 复数全连接层
        ComplexLinear(256),
        ComplexReLU(),
        # 输出层
        ComplexLinear(num_classes),
        TakeReal(),  # 只取实部用于分类
        tf.keras.layers.Softmax()
    ])
    return model


# 9. 自定义训练步骤
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


# 10. 主训练循环
def train_complex_cnn():
    # 加载数据
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 预处理数据
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # 添加通道维度
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # 转换为one-hot编码
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # 创建模型
    model = create_complex_cnn_model()

    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # 训练参数
    epochs = 5
    batch_size = 512

    # 训练循环
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # 打乱训练数据
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        # 分批训练
        epoch_loss = 0
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            loss = train_step(model, x_batch, y_batch, optimizer, loss_fn)
            epoch_loss += loss

            if i % 1000 == 0:
                print(f"  Batch {i}, Loss: {loss:.4f}")

        # 评估模型
        test_loss = loss_fn(y_test, model.predict(x_test, verbose=0))
        test_acc = tf.keras.metrics.CategoricalAccuracy()(y_test, model.predict(x_test, verbose=0))
        print(
            f"Epoch Loss: {epoch_loss / (len(x_train) / batch_size):.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return model


# 11. 运行训练
if __name__ == "__main__":
    model = train_complex_cnn()