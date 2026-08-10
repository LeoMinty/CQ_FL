# 导入数据集
import tensorflow.keras.datasets.mnist as mnist
# 绘图用
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers, regularizers, constraints, activations
import numpy as np

B_S = 32

def simplified_complex_matmul(A, B):
    """
    简化复数矩阵乘法 C = AB，其中 B 的元素仅来自 {1, -1, i, -i}。
    内存安全版本：用分离掩码 + matmul，避免 tile 展开导致 OOM。

    参数:
        A (tf.Tensor): 形状 (m, n, 2)
        B (tf.Tensor): 形状 (n, k, 2)，元素仅来自 {1, -1, i, -i}
    返回:
        tf.Tensor: 形状 (m, k, 2)
    """
    # 分离实虚部
    A_real, A_imag = A[..., 0], A[..., 1]  # (m, n)
    B_real, B_imag = B[..., 0], B[..., 1]  # (n, k)

    # 四个掩码：B 的四个可能值
    is_one     = tf.cast((B_real ==  1.0) & (B_imag ==  0.0), tf.float32)  # (n, k)
    is_neg_one = tf.cast((B_real == -1.0) & (B_imag ==  0.0), tf.float32)
    is_i       = tf.cast((B_real ==  0.0) & (B_imag ==  1.0), tf.float32)
    is_neg_i   = tf.cast((B_real ==  0.0) & (B_imag == -1.0), tf.float32)

    # 复数乘法: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    # 对于 B={1,-1,i,-i}:
    #   B=1:     (a, b)  → C_real += a,    C_imag += b
    #   B=-1:    (-a, -b) → C_real -= a,    C_imag -= b
    #   B=i:     (-b, a)  → C_real -= b,    C_imag += a
    #   B=-i:    (b, -a)  → C_real += b,    C_imag -= a
    #
    # 用 matmul 实现: C = A @ mask

    C_real = (tf.matmul(A_real, is_one) - tf.matmul(A_real, is_neg_one)
              - tf.matmul(A_imag, is_i) + tf.matmul(A_imag, is_neg_i))

    C_imag = (tf.matmul(A_imag, is_one) - tf.matmul(A_imag, is_neg_one)
              + tf.matmul(A_real, is_i) - tf.matmul(A_real, is_neg_i))

    return tf.stack([C_real, C_imag], axis=-1)

class ComplexConv2D(Layer):
    """
    改进的复数卷积层，包含STE优化和量化正则化
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 quant_weight_decay=0.01,  # 量化正则化强度
                 use_quant=True,          # True=量化, False=fp32卷积
                 quantize_backward=True, # True=2-bit相位梯度, False=标准STE
                 **kwargs):
        super(ComplexConv2D, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        self.strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.quant_weight_decay = quant_weight_decay  # 量化正则化系数
        self.use_quant = use_quant
        self.quantize_backward = quantize_backward

    def build(self, input_shape):
        # 输入形状: (batch_size, height, width, in_channels, 2)
        input_channels = input_shape[3]  # 输入通道数

        # 复数卷积核形状: [kernel_h, kernel_w, in_channels, filters, 2]
        kernel_shape = self.kernel_size + (input_channels, self.filters, 2)

        # 创建复数卷积核权重
        self.kernel_fp = self.add_weight(
            name='kernel_fp',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True
        )

        # 如果使用偏置，创建复数偏置权重
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters, 2),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True
            )
        else:
            self.bias = None

        super(ComplexConv2D, self).build(input_shape)

    def call(self, inputs):
        """
        执行复数卷积操作，可选量化
        """
        if self.use_quant:
            kernel = self.phase_quant_with_ste(self.kernel_fp)
            # 量化模式：用 simplified_complex_matmul（无乘法，但仅在量化后可用）
            outputs = self._vectorized_conv2d(inputs, kernel)
        else:
            # 无量化模式：用标准可微 conv2d 实现复数卷积
            outputs = self._differentiable_conv2d(inputs, self.kernel_fp)

        outputs_real = outputs[..., 0]
        outputs_imag = outputs[..., 1]

        # 添加复数偏置
        if self.use_bias:
            outputs_real += self.bias[None, None, None, :, 0]
            outputs_imag += self.bias[None, None, None, :, 1]

        outputs = tf.stack([outputs_real, outputs_imag], axis=-1)

        # 应用激活函数
        if self.activation is not None:
            # 分别对实部和虚部应用激活函数
            real_activated = self.activation(outputs[..., 0])
            imag_activated = self.activation(outputs[..., 1])
            outputs = tf.stack([real_activated, imag_activated], axis=-1)

        # 添加量化正则化损失
        if self.use_quant:
            self.add_quantization_regularization()

        return outputs

    def phase_quant_with_ste(self, weights):
        """
        前向：权重量化到 {1, i, -1, -i}
        反向：梯度也量化到 {1, i, -1, -i}（LPT-FL 风格）
        """
        @tf.custom_gradient
        def _op(x):
            quantized = self.phase_quant(x)

            def grad_fn(grad):
                if self.quantize_backward:
                    # CQ-FL/原BitFL路径：反向梯度量化到四个复数主轴。
                    return self.phase_quant(grad)
                # 2-bit W + FP32 Adam消融：前向仍量化，反向使用标准STE。
                return grad

            return quantized, grad_fn

        return _op(weights)

    def phase_quant(self, complex_weights):
        """
        将复数权重量化为 {1, i, -1, -i} 四个值之一
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

    def add_quantization_regularization(self):
        """
        添加量化正则化损失，促使权重逼近离散值
        """
        if self.quant_weight_decay > 0:
            # 计算量化权重
            quantized = self.phase_quant(self.kernel_fp)

            # 计算全精度权重与量化权重之间的L2距离
            quantization_loss = tf.reduce_mean(tf.square(self.kernel_fp - quantized))

            # 添加正则化损失
            self.add_loss(self.quant_weight_decay * quantization_loss)

    def _vectorized_conv2d(self, inputs, kernel):
        """使用向量化操作实现高效的2D卷积"""
        # 获取卷积核尺寸
        kernel_height, kernel_width = kernel.shape[0], kernel.shape[1]
        in_channels = inputs.shape[-2]
        out_channels = self.filters

        inputs_real = inputs[..., 0]
        inputs_imag = inputs[..., 1]

        # 使用tf.image.extract_patches高效提取滑动窗口
        patches_real = tf.image.extract_patches(
            inputs_real,
            sizes=[1, kernel_height, kernel_width, 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper()
        )
        patches_imag = tf.image.extract_patches(
            inputs_imag,
            sizes=[1, kernel_height, kernel_width, 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding.upper()
        )

        patches = tf.stack([patches_real, patches_imag], axis=-1)

        # 获取提取的patch的形状
        patches_shape = tf.shape(patches)
        batch_size = patches_shape[0]
        out_height = patches_shape[1]
        out_width = patches_shape[2]
        patch_size = kernel_height * kernel_width * in_channels

        # 重塑patches为[batch_size * out_height * out_width, patch_size]
        patches_reshaped = tf.reshape(patches, [batch_size * out_height * out_width, patch_size, 2])

        # 重塑卷积核为[patch_size, out_channels]
        kernel_reshaped = tf.reshape(kernel, [patch_size, out_channels, 2])

        outputs = simplified_complex_matmul(patches_reshaped, kernel_reshaped)

        outputs = tf.reshape(outputs, [batch_size, out_height, out_width, out_channels, 2])

        return outputs

    def _differentiable_conv2d(self, inputs, kernel):
        """
        标准可微复数卷积：用 tf.nn.conv2d 实现 (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
        用于 use_quant=False 模式，保证梯度流正常
        """
        real_in, imag_in = inputs[..., 0], inputs[..., 1]
        kr, ki = kernel[..., 0], kernel[..., 1]

        real_real = tf.nn.conv2d(real_in, kr, strides=self.strides, padding=self.padding.upper())
        imag_imag = tf.nn.conv2d(imag_in, ki, strides=self.strides, padding=self.padding.upper())
        real_imag = tf.nn.conv2d(real_in, ki, strides=self.strides, padding=self.padding.upper())
        imag_real = tf.nn.conv2d(imag_in, kr, strides=self.strides, padding=self.padding.upper())

        out_real = real_real - imag_imag
        out_imag = real_imag + imag_real
        return tf.stack([out_real, out_imag], axis=-1)

    def compute_output_shape(self, input_shape):
        # 输入形状: (batch_size, in_height, in_width, in_channels, 2)
        batch_size, in_height, in_width, in_channels, _ = input_shape
        kernel_height, kernel_width = self.kernel_size

        # 计算输出空间维度
        if self.padding == 'same':
            out_height = (in_height + self.strides[0] - 1) // self.strides[0]
            out_width = (in_width + self.strides[1] - 1) // self.strides[1]
        else:  # 'valid'
            out_height = (in_height - kernel_height) // self.strides[0] + 1
            out_width = (in_width - kernel_width) // self.strides[1] + 1

        return (batch_size, out_height, out_width, self.filters, 2)

    def get_config(self):
        config = super(ComplexConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': activations.serialize(self.activation) if self.activation else None,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'quant_weight_decay': self.quant_weight_decay,
            'use_quant': self.use_quant,
            'quantize_backward': self.quantize_backward,
        })
        return config


class ComplexBatchNormalization(Layer):
    def __init__(self, **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.bn_real = tf.keras.layers.BatchNormalization()
        self.bn_imag = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        real = self.bn_real(inputs[..., 0], training=training)
        imag = self.bn_imag(inputs[..., 1], training=training)
        return tf.stack([real, imag], axis=-1)

if __name__ == "__main__":
    '''
    # 创建一个简单的测试
    print("测试高效自定义卷积层...")

    # 创建随机输入
    input_tensor = tf.random.normal([2, 84, 84, 3])  # 使用错误信息中的尺寸
    print("输入形状:", input_tensor.shape)

    # 创建高效自定义卷积层
    conv_layer = EfficientCustomConv2D(filters=64, kernel_size=3, padding='same')

    # 应用卷积层
    output_tensor = conv_layer(input_tensor)
    print("输出形状:", output_tensor.shape)

    # 与TensorFlow内置卷积层比较
    print("\n与TensorFlow内置卷积层比较...")
    tf_conv_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same')
    tf_output_tensor = tf_conv_layer(input_tensor)
    print("TensorFlow输出形状:", tf_output_tensor.shape)

    # 性能测试
    print("\n性能测试...")
    import time

    # 测试高效实现
    start = time.time()
    for _ in range(10):
        _ = conv_layer(input_tensor)
    end = time.time()
    print(f"高效实现耗时: {end - start:.4f}秒")'''

    # 测试三重循环实现（仅用于比较，实际会非常慢）
    # 警告：对于大输入，这会非常慢！
    # print("\n测试三重循环实现...")
    # slow_conv_layer = CustomConv2D(filters=64, kernel_size=3, padding='same')
    # start = time.time()
    # for _ in range(10):
    #     _ = slow_conv_layer(input_tensor)
    # end = time.time()
    # print(f"三重循环实现耗时: {end - start:.4f}秒")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化
    # 数据
    x_train = tf.expand_dims(x_train, -1)  # 卷积的输入一般是一个四维的数据，还需要一个“通道“ ，因此在最后扩展一个维度
    x_test = tf.expand_dims(x_test, -1)

    # 标签
    y_train = np.float32(tf.keras.utils.to_categorical(y_train, num_classes=10))  # one-hot处理
    y_test = np.float32(tf.keras.utils.to_categorical(y_test, num_classes=10))

    batch_size = B_S
    # 创建tf Dataset数据集，进行数据打包，方便地组合成train和label的配对数据集
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size).shuffle(batch_size * 10)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    from Bit2Conv import ComplexMaxPool2D
    from Bit2Linear import ComplexLinear, RealToComplex, TakeReal
    # 模型创建，采用函数式进行编程
    # 创建复数卷积模型
    model = tf.keras.Sequential([
        # 输入层: 28x28 灰度图像
        tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),

        # 转换为复数表示: 实部=原图，虚部=0
        RealToComplex(),

        # 复数卷积层
        ComplexConv2D(filters=16, kernel_size=3, padding='same', activation=tf.nn.relu),
        ComplexBatchNormalization(),

        # 复数卷积层
        ComplexConv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu),

        # 复数池化层
        ComplexMaxPool2D(pool_size=(2, 2)),

        # 另一个复数卷积层
        ComplexConv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu),

        # 复数池化层
        ComplexMaxPool2D(pool_size=(2, 2)),

        # 展平
        tf.keras.layers.Reshape((3136, 2), input_shape=(7, 7, 64, 2)),

        # 只取实部
        TakeReal(),

        # 全连接层
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    # 模型编译  也可以optimizer=tf.optimizers.Adam(1e-3), loss=tf.losses.categorical_crossentropy, metrics = ['accuracy']
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    # 模型训练5个epochs
    model.fit(x_train, y_train, batch_size=batch_size, epochs=5)
    # 模型保存本地
    # model.save("./saver/MyTfModelForMnist.h5")
    # 模型在测试集上的评估
    score = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("测试集准确率:", score)  # 输出 [损失率，准确率]
