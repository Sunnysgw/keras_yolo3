# coding: utf-8
from functools import wraps
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Add, UpSampling2D, Concatenate
from keras.regularizers import l2
from keras.models import Model
from functools import reduce


def compose(*funcs):
    """
    把输入的层给链接在一起
    :param funcs:要连接在一起的网络层
    :return:
    """
    # 使用reduce来实现
    if funcs:
        return lambda x: reduce(lambda v, f: f(v), funcs, x)
    else:
        raise ValueError('传进来的list为空')


@wraps(Conv2D)
def dark_net_conv(*args, **kwargs):
    """
    dark_net中用到的卷积模块
    wraps的意义，使用自己的方式对Conv2D进行封装，调用这个函数的方式和Conv2D一致，参数也是一致
    :return:
    """
    # 自己的dark_net参数
    dark_net_kwargs = dict()
    # 设置kernel的初始化方式，使用l2正则化方式初始化
    dark_net_kwargs['kernel_regularizer'] = l2(5e-4)
    # 设置填充方式，当步长是(2, 2)的时候不填充，长宽分别减半，正常是使用same，保证输出尺寸不变
    dark_net_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    # 把传进来的参数更新进来
    dark_net_kwargs.update(kwargs)
    # 使用更新好的参数调用卷积层
    return Conv2D(*args, **dark_net_kwargs)


def dark_net_conv_bn_leakey(*args, **kwargs):
    """
    dark_net中的卷积模块，包括卷积层、bn层、激活函数
    :param args:
    :param kwargs:
    :return:
    """
    # 不使用偏置项
    no_bias_kwargs = dict()
    no_bias_kwargs['use_bias'] = False
    no_bias_kwargs.update(kwargs)
    return compose(dark_net_conv(*args, **no_bias_kwargs),
                   BatchNormalization(),
                   LeakyReLU(alpha=0.1))


def res_block(x, num_filter, num_block):
    """
    dark_net中的残差模块
    :param x:输入tensor
    :param num_filter:卷积层的输出维度
    :param num_block:其中的残差块的个数
    :return:
    """
    # 分别在分别在高度和宽度上填充一维，这样保证在下面的步长为2的卷积层中刚好把维度减半
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = dark_net_conv_bn_leakey(num_filter, (3, 3), strides=(2, 2))(x)
    for _ in range(num_block):
        y = compose(dark_net_conv_bn_leakey(num_filter//2, (1, 1)),
                    dark_net_conv_bn_leakey(num_filter, (3, 3)))(x)
        x = Add()([x, y])
    return x


def dark_net_body(x):
    """
    yolo的基础神经网络，darknet-53
    :param x:输入tensor
    :return:
    """
    x = dark_net_conv_bn_leakey(32, (3, 3))(x)
    x = res_block(x, 64, 1)
    x = res_block(x, 128, 2)
    x = res_block(x, 256, 8)
    x = res_block(x, 512, 8)
    x = res_block(x, 1024, 4)
    return x


def make_last_layers(x, num_filter, out_filter):
    """
    把dark_net的输出封装成正经的输出
    :param x:
    :param num_filter:
    :param out_filter:
    :return:
    """
    x = compose(dark_net_conv_bn_leakey(num_filter, (1, 1)),
                dark_net_conv_bn_leakey(num_filter*2, (3, 3)),
                dark_net_conv_bn_leakey(num_filter, (1, 1)),
                dark_net_conv_bn_leakey(num_filter*2, (3, 3)),
                dark_net_conv_bn_leakey(num_filter, (1, 1)))(x)
    y = compose(dark_net_conv_bn_leakey(num_filter*2, (3, 3)),
                dark_net_conv(out_filter, (1, 1)))(x)
    return x, y


def yolo_body(input_, num_anchors, num_classes):
    """
    基于dark_net实现整个yolo网络
    :param input_:
    :param num_anchors:
    :param num_classes:
    :return:
    """
    # 首先构建dark_net
    dark_net = Model(input_, dark_net_body(input_))
    # 下面就使用一样的套路构建三个输出层
    # 构建第一层，奇怪，为什么这里不直接分开呢，而是乘起来
    x, y1 = make_last_layers(dark_net.output, 512, num_anchors*(5+num_classes))
    # 上一层经过(1, 1)卷积，然后上采样
    x = compose(dark_net_conv_bn_leakey(256, (1, 1)),
                UpSampling2D(2))(x)
    # 把x和倒数第二个残差块的输出拼接在一起
    x = Concatenate()([x, dark_net.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors*(5+num_classes))
    # 上一层经过(1, 1)卷积，然后上采样
    x = compose(dark_net_conv_bn_leakey(128, (1, 1)),
                UpSampling2D(2))(x)
    # 把x和倒数第二个残差块的输出拼接在一起
    x = Concatenate()([x, dark_net.layers[92].output])
    _, y3 = make_last_layers(x, 128, num_anchors*(5+num_classes))
    # 返回整个model
    return Model(input_, [y1, y2, y3])
