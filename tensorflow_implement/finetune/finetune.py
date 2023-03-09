import numpy as np
np.random.seed(1333337)
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras import backend as K
from multi import MultiHeadAttention
from keras_contrib.layers import InstanceNormalization
import os
from multi import MultiHeadAttention
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 512
num_classes = 4
epochs = 10

num_codes = 64
num_layers = 3
z_dim = 32

img_rows, img_cols = 32, 128
input_shape = (img_rows, img_cols, 1)
img_rows, img_cols = 32, 128

train_data = pickle.load(open("data/iemocap_vad/spectrograms/cross_flod1/train.pkl", "rb"))
valid_data = pickle.load(open("data/iemocap_vad/spectrograms/cross_flod1/valid.pkl", "rb"))
saved_feature_path = "data/iemocap_vad/spectrograms/cross_flod1/pretrain_feat.pkl"

x_train = train_data["spectrograms"]
y_train = train_data["labels"]

x_valid = train_data["spectrograms"]
y_valid = valid_data["labels"]

# 打乱数据集
index = np.arange(len(x_train))
np.random.shuffle(index) # 打乱索引
x_train, y_train = x_train[index], y_train[index]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'spec samples')


def resnet_block(x):
    dim = K.int_shape(x)[-1]
    xo = x
    x = Conv2D(dim, 3, padding='same', activation='relu')(x)
    x = Conv2D(dim, 1, padding='same', activation='relu')(x)
    return Add()([xo, x])


# encoder
x_in = Input(shape=input_shape)
x_en = x_in

x_en = Conv2D(filters=z_dim, kernel_size=4, strides=2, padding='same', activation='relu', name='100')(x_en)
x_en = Conv2D(z_dim, 4, strides=2, padding='same', activation='relu', name='101')(x_en)
x_en = Conv2D(z_dim, 4, strides=2, padding='same', activation='relu', name='102')(x_en)
x_en = Conv2D(z_dim, 4, strides=2, padding='same', activation='relu', name='103')(x_en)
x_en = BatchNormalization()(x_en)

for i in range(num_layers):
    x_en = resnet_block(x_en)
    if i < num_layers - 1:
        x_en = BatchNormalization()(x_en)

e_model = Model(x_in, x_en)
e_model.summary()

# decoder


# ENR
ENR_in = Input(shape=K.int_shape(x_en)[1:])
F = ENR_in
F_hat = InstanceNormalization()(F)

R = Subtract()([F, F_hat])
R = Reshape((16, 32))(R)
# R = K.reshape(R, [-1, 16, 32])

r = MultiHeadAttention(head_num=8, name='Multi-Head')(R)
r = Reshape((2, 8, 32))(r)

f = Add()([F_hat, r])
ENR_model = Model(ENR_in, f)  # Emotion Label and Output

ENR_model.summary()

# vq
vq_in = Input(shape=K.int_shape(f)[1:])
vq = vq_in


class VectorQuantizer(Layer):
    """vq
    """

    def __init__(self, num_codes, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.num_codes = num_codes

    def build(self, input_shape):
        super(VectorQuantizer, self).build(input_shape)
        dim = input_shape[-1]
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.num_codes, dim),
            initializer='uniform'
        )

    def call(self, inputs):
        """inputs.shape=[None, m, m, dim]
        """
        l2_inputs = K.sum(inputs ** 2, -1, keepdims=True)
        l2_embeddings = K.sum(self.embeddings ** 2, -1)
        for _ in range(K.ndim(inputs) - 1):
            l2_embeddings = K.expand_dims(l2_embeddings, 0)
        embeddings = K.transpose(self.embeddings)
        dot = K.dot(inputs, embeddings)
        distance = l2_inputs + l2_embeddings - 2 * dot
        codes = K.cast(K.argmin(distance, -1), 'int32')
        code_vecs = K.gather(self.embeddings, codes)
        return code_vecs

    def compute_output_shape(self, input_shape):
        return input_shape


vq_layer = VectorQuantizer(num_codes)
code_vecs = vq_layer(vq)

vq_model = Model(vq_in, code_vecs)
vq_model.summary()

# decoder
d_in = Input(shape=K.int_shape(code_vecs)[1:])
x_de = d_in

for i in range(num_layers):
    x_de = BatchNormalization()(x_de)
    x_de = resnet_block(x_de)

x_de = Conv2DTranspose(z_dim, 4, strides=2, padding='same', activation='relu', name='110')(x_de)
x_de = Conv2DTranspose(z_dim, 4, strides=2, padding='same', activation='relu', name='120')(x_de)
x_de = Conv2DTranspose(z_dim, 4, strides=2, padding='same', activation='relu', name='130')(x_de)
x_de = Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh', name='140')(x_de)

d_model = Model(d_in, x_de)
d_model.summary()

# train
x_in = Input(shape=input_shape)
x = x_in

en_out = e_model(x)
ENR_out = ENR_model(en_out)

ENR_out1 = Conv2D(64, kernel_size=(1, 2), activation='relu')(ENR_out)
ENR_out1 = Conv2D(64, kernel_size=(1, 2), activation='relu')(ENR_out1)
ENR_out1 = Flatten()(ENR_out1)

label_out = Dense(4, activation="softmax", name='emo')(ENR_out1)
vq_out = vq_model(ENR_out)
ENR_vq = Lambda(lambda x: x[0] + K.stop_gradient(x[1] - x[0]))([ENR_out, vq_out])
# new add
ENR_vq_flatten = Flatten(name="ENR_vq_flatten")(ENR_vq)

de_out = d_model(ENR_vq)

train_model = Model(x_in, [label_out, de_out])

mse_x = K.mean((x_in - de_out) ** 2)
mse_e = K.mean((K.stop_gradient(ENR_out) - vq_out) ** 2)  # ???
mse_z = K.mean((K.stop_gradient(vq_out) - ENR_out) ** 2)
loss1 = mse_x + mse_e + 0.25 * mse_z

train_model.add_loss(loss1)
train_model.compile(optimizer=Adam(1e-3),
                    loss={'emo': keras.losses.categorical_crossentropy},
                    loss_weights={'emo': 0.5})
train_model.summary()



train_model.fit(x_train, [y_train, x_train],
                epochs=40,
                verbose=1,
                batch_size=batch_size,
                )

pretrain_model_path = os.path.split(__file__)[0] + "/prertain_model.hdf5"
if os.path.isfile(pretrain_model_path):
    train_model.load_weights(pretrain_model_path)

# get feature
intermediate_layer_model =Model(inputs=x_in,
                                 outputs=train_model.get_layer(name="ENR_vq_flatten").output)

train_output = intermediate_layer_model.predict(x_train)
valid_output = intermediate_layer_model.predict(x_valid)

# save features


pickle.dump({
            'train_data':train_output,
            'valid_data':valid_output,
            },
            open(feature_path, "wb"))