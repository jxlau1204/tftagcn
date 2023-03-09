#! -*- coding: utf-8 -*-
# refer: https://kexue.fm/archives/5112
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import backend as K
# from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer, Conv2D, Dense
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.ops import gen_math_ops, standard_ops
from channel_attention import *

__all__ = ["Capsule", "squash"]

def squash(x, axis=-1, name_index=None):
    x = channel_attention(x, x.shape[-1], reduction=16, name_index=name_index)
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    squash_temp = scale * x
    output = K.permute_dimensions(squash_temp, (0, 2, 3, 1))
    # return scale * x
    return output

# def squash(x, axis=-1):
#     s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
#     scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
#     return scale * x


# define our own softmax function instead of K.softmax
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule=10, dim_capsule=(60, 16), routings=3, share_weights=True, activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

        self.kernel_initializer = initializers.get('glorot_uniform')
        self.kernel_regularizer = regularizers.get(None)
        self.bias_regularizer = regularizers.get(None)
        self.kernel_constraint = constraints.get(None)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[1] * input_shape[2]  # (60, 16)
        input_channel = input_shape[-1]
        if self.share_weights:  # self.W shape: [1， 128， 10 * 16]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(2, 2, input_channel, input_channel),
                                     initializer='glorot_uniform',
                                     trainable=True
                                     )
            self.kernel = self.add_weight(
                'kernel',
                shape=[input_dim_capsule, self.num_capsule * self.dim_capsule[0] * self.dim_capsule[1]],
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                dtype=self.dtype,
                trainable=True)
            # self.dense = self.add_weight(name="dense_kernel",
            #                              shape=()
            #                              )
            # self.dense = self.add_weight(name="dense_layer",
            #                              shape=())
        # else:
        #     input_num_capsule = input_shape[-2]
        #     self.W = self.add_weight(name='capsule_kernel',
        #                              shape=(input_num_capsule,
        #                                     input_dim_capsule,
        #                                     self.num_capsule * self.dim_capsule),
        #                              initializer='glorot_uniform',
        #                              trainable=True)
    def get_config(self):
        config = {
            'num_capsule': self.num_capsule,
            'dim_capsule': self.dim_capsule,
            'share_weights': self.share_weights,
            'routings': self.routings,
            'activation': keras.activations.serialize(self.activation),
            "kernel_initializer": self.kernel_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
            "kernel_constraint": self.kernel_constraint
        }
        base_config = super(Capsule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, u_vecs):  # u_vecs (10, 60, 16, 64)
        # tf.print("***************************")
        batch_size = K.shape(u_vecs)[0]  # 10
        input_num_capsule = K.shape(u_vecs)[-1]  # 64
        if self.share_weights:
            # u_hat_vecs = K.conv1d(u_vecs, self.W)
            # u_hat_vecs (10, 60, 16, 64)
            # u_hat_vecs = Conv2D(filters=64, kernel_size=(2,2), padding="same", data_format="channels_last")(u_vecs)
            u_hat_vecs = K.conv2d(u_vecs, self.W, padding="same", data_format="channels_last")
            # tf.print("u_hat_vecs:", u_hat_vecs.shape)
            u_hat_vecs_temp = K.reshape(u_hat_vecs, (batch_size, input_num_capsule, K.shape(u_hat_vecs)[1] * K.shape(u_hat_vecs)[2]))
            # tf.print("u_hat_vecs_temp:", u_hat_vecs_temp.shape)
            # tf.print("....:", self.num_capsule * self.dim_capsule[0] * self.dim_capsule[1])

            rank = u_hat_vecs_temp.shape.rank
            if rank is not None and rank > 2:
                # Broadcasting is required for the inputs.
                outputs = standard_ops.tensordot(u_hat_vecs_temp, self.kernel, [[rank - 1], [0]])
                # Reshape the output back to the original ndim of the input.
                if not context.executing_eagerly():
                    shape = u_hat_vecs_temp.shape.as_list()
                    output_shape = shape[:-1] + [self.num_capsule * self.dim_capsule[0] * self.dim_capsule[1]]
                    outputs.set_shape(output_shape)
            else:
                outputs = gen_math_ops.mat_mul(u_hat_vecs_temp, self.kernel)
            u_hat_vecs = outputs

        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        # batch_size = K.shape(u_vecs)[0] # 10
        # input_num_capsule = K.shape(u_vecs)[-1]  # 64
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule[0],
                                            self.dim_capsule[1]))  # (10, 64, 6, 60, 16)
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3, 4))  # (10, 6, 64, 60, 16)
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        # [10, 6, 64]
        b = K.zeros_like(u_hat_vecs[:, :, :, 0, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)  # [10, 6, 64]
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binjk->bijk', c, u_hat_vecs)  # [10, 6, 60, 16]
            # [10, 6, 960]
            o_temp = K.reshape(o, shape=(batch_size, self.num_capsule, self.dim_capsule[0] * self.dim_capsule[1]))
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o_temp = K.l2_normalize(o_temp, -1)
                # o [10, 6, 60, 16]
                o = K.reshape(o_temp, shape=(batch_size, self.num_capsule, self.dim_capsule[0], self.dim_capsule[1]))
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bijk,binjk->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        # squash_temp = self.activation(o)
        # output = K.permute_dimensions(squash_temp, (0, 2, 3, 1))
        # return output
        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
