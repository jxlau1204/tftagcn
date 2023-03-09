#! /Work18/.../conda3_env/env_1 python
# @Time : 2021/3/12 16:18
# @Author : syd
# @File : channel_attention.py
import tensorflow as tf
# from tensorflow.keras.layers import
from tensorflow.keras import backend as K

__all__ = ["channel_attention"]


reg = 5e-4
# w_reg = tf.nn.l2_regularizer(reg)
# # w_init = tf.VarianceScaling(factor=1., mode='FAN_AVG', uniform=True)
# w_init = tf.nn.variance_scaling_initializer(factor=1., mode='FAN_AVG', uniform=True)
# b_init = tf.zeros_initializer()



def adaptive_global_average_pool_2d(x):
    """
    In the paper, using gap which output size is 1, so i just gap func :)
    :param x: 4d-tensor, (batch_size, height, width, channel)
    :return: 4d-tensor, (batch_size, 1, 1, channel)
    """
    c = x.get_shape()[-1]
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

def conv2d(x, f=64, k=3, s=1, pad='SAME', use_bias=True, reuse=None, name='conv2d'):
    """
    :param x: input
    :param f: filters
    :param k: kernel size
    :param s: strides
    :param pad: padding
    :param use_bias: using bias or not
    :param reuse: reusable
    :param name: scope name
    :return: output
    """
    layer = tf.keras.layers.Conv2D(filters=f, kernel_size=k, strides=s,
                                   # kernel_initializer=w_init,
                                   # kernel_regularizer=w_reg,
                                   # bias_initializer=b_init,
                                   padding=pad,
                                   use_bias=use_bias,
                                   # reuse=reuse,
                                   name=name)
    return layer(x)
    # return tf.keras.layers.Conv2D(inputs=x,
    # # return tf.compat.v1.layers.conv2d(inputs=x,
    #                         filters=f, kernel_size=k, strides=s,
    #                         # kernel_initializer=w_init,
    #                         # kernel_regularizer=w_reg,
    #                         # bias_initializer=b_init,
    #                         padding=pad,
    #                         use_bias=use_bias,
    #                         reuse=reuse,
    #                         name=name)



def channel_attention(x, f, reduction, name_index):
    """
    Channel Attention (CA) Layer
    :param x: input layer
    :param f: conv2d filter size
    :param reduction: conv2d filter reduction rate
    :param name: scope name
    :return: output layer
    """
    # with tf.variable_scope("CA-%s" % name):
    skip_conn = tf.identity(x, name=f'identity-{name_index}')

    x = adaptive_global_average_pool_2d(x)

    x = conv2d(x, f=f // reduction, k=1, name=f"conv2d-1-{name_index}")
    x = K.relu(x)

    x = conv2d(x, f=f, k=1, name=f"conv2d-2-{name_index}")
    # x = tf.nn.sigmoid(x)
    x = K.sigmoid(x)
    return tf.multiply(skip_conn, x)



