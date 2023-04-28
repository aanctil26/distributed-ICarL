import tensorflow as tf
import numpy as np
import pickle as cPickle

class Conv(tf.keras.layers.Layer):
    def __init__(self, size, out_channels, strides=(1, 1),
                 dilation=None, padding='SAME', apply_relu=True, alpha=0.0,
                 bias=True, initializer=None, name=None, **kwargs):
        super(Conv, self).__init__(name=name, **kwargs)
        self.size = size
        self.out_channels = out_channels
        self.strides = strides
        self.dilation = dilation
        self.padding = padding
        self.apply_relu = apply_relu
        self.alpha = alpha
        self.bias = bias
        self.initializer = initializer or tf.keras.initializers.GlorotUniform()

    def build(self, input_shape):
        in_channels = input_shape[-1]
        self.W = self.add_weight("W", shape=[self.size, self.size, in_channels, self.out_channels], dtype=tf.float32,
                                 initializer=self.initializer, regularizer=tf.keras.regularizers.l2(0.5))
        if self.bias:
            self.b = self.add_weight("b", shape=[1, 1, 1, self.out_channels], dtype=tf.float32,
                                     initializer=tf.zeros_initializer())
        super(Conv, self).build(input_shape)

    def call(self, inp):
        if self.dilation:
            assert self.strides == (1, 1)
            out = tf.add(tf.nn.atrous_conv2d(inp, self.W, rate=self.dilation, padding=self.padding), self.b, name='convolution')
        else:
            out = tf.add(tf.nn.conv2d(inp, self.W, strides=[1, *self.strides, 1], padding=self.padding), self.b, name='convolution')

        if self.apply_relu:
            out = tf.nn.relu(out) if self.alpha == 0 else tf.nn.leaky_relu(out, alpha=self.alpha)
        return out


def call(self, inp):
    layer = self.resconv1(inp)
    layer = self.batch_norm_resconv1(layer, training=self.phase == 'train')
    layer = self.resconv2(layer)
    layer = self.batch_norm_resconv2(layer, training=self.phase == 'train')

    if self.increase_dim:
        projection = self.projconv(inp)
        projection = self.batch_norm_projconv(projection, training=self.phase == 'train')
        block = layer + projection
    else:
        block = layer + inp

    if not self.last:
        block = tf.nn.relu(block)

    return block

def ResNet18(inp, phase, num_outputs=1000, alpha=0.0):
    def residual_block(inp, phase, alpha=0.0, nom='a', increase_dim=False, last=False):
        input_num_filters = inp.shape[-1]
        if increase_dim:
            first_stride = (2, 2)
            out_num_filters = input_num_filters * 2
        else:
            first_stride = (1, 1)
            out_num_filters = input_num_filters

        layer = Conv2D(out_num_filters, 3, strides=first_stride, padding='same', activation=None)(inp)
        layer = BatchNormalization()(layer)
        layer = ReLU()(layer)
        layer = Conv2D(out_num_filters, 3, strides=(1, 1), padding='same', activation=None)(layer)
        layer = BatchNormalization()(layer)

        if increase_dim:
            projection = Conv2D(out_num_filters, 1, strides=(2, 2), padding='same', activation=None)(inp)
            projection = BatchNormalization()(projection)
        else:
            projection = inp

        if last:
            block = Add()([layer, projection])
        else:
            block = Add()([layer, projection])
            block = ReLU()(block)

        return block

    # First conv
    layer = Conv2D(64, 7, strides=(2, 2), padding='same', activation=None)(inp)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(layer)

    # First stack of residual blocks
    for letter in 'ab':
        layer = residual_block(layer, phase, alpha=0.0, nom=letter)

    # Second stack of residual blocks
    layer = residual_block(layer, phase, alpha=0.0, nom='c', increase_dim=True)
    for letter in 'd':
        layer = residual_block(layer, phase, alpha=0.0, nom=letter)

    # Third stack of residual blocks
    layer = residual_block(layer, phase, alpha=0.0, nom='e', increase_dim=True)
    for letter in 'f':
        layer = residual_block(layer, phase, alpha=0.0, nom=letter)

    # Fourth stack of residual blocks
    layer = residual_block(layer, phase, alpha=0.0, nom='g', increase_dim=True)
    layer = residual_block(layer, phase, alpha=0.0, nom='h', increase_dim=False, last=True)

    layer = GlobalAveragePooling2D()(layer)
    layer = Dense(num_outputs, activation=None)(layer)

    return layer


def get_weight_initializer(params):
    initializer = []
    for layer, value in params.items():
        var = tf.compat.v1.get_variable('%s' % layer)
        op = var.assign(value)
        initializer.append(op)
    return initializer


def save_model(name, scope, sess):
    variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.WEIGHTS, scope=scope)
    d = [(v.name.split(':')[0], sess.run(v)) for v in variables]
    cPickle.dump(d, open(name, 'wb'))
