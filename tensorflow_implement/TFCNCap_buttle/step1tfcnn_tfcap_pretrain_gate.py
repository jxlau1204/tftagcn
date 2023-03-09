import numpy as np
import random
import os
import tensorflow as tf
import scipy.io as sio
import pickle
seed = 1333337

def seed_tensorflow(seed=1333337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_tensorflow(seed)

from tensorflow import keras
from numpy import linalg as LA
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
from keras.models import *
from keras.callbacks import *
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN, LSTM, GRU, Reshape, BatchNormalization, \
    Concatenate
from keras.layers import (Input, Embedding, Dense, Conv1D, MaxPool1D, Flatten,
                          Dropout, concatenate, multiply, RepeatVector, Lambda)
from keras.layers import Bidirectional
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K
import h5py
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# from multi import MultiHeadAttention, Capsule
import scipy.io as sio 
from keras.utils import np_utils
from my_capsule_keras import *
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if tf.__version__.startswith('1.'):  # tensorflow 1
    config = tf.ConfigProto()  # allow_soft_placement=True
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
else:  # tensorflow 2
    physical_devices = tf.config.list_physical_devices('GPU')
    print("physical_devices:", physical_devices, flush=True)
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

batch_size = 128
num_classes = 4
epochs = 20
batch_index = 0
batch_size = 256

# input image dimensions
img_rows, img_cols = 32, 128

train_data = pickle.load(open("data/iemocap_vad/spectrograms/cross_flod1/train.pkl", "rb"))
valid_data = pickle.load(open("data/iemocap_vad/spectrograms/cross_flod1/valid.pkl", "rb"))
finetune_data = pickle.load(open("data/iemocap_vad/spectrograms/cross_flod1/pretrain_feat.pkl", "rb"))

x_train = train_data["spectrograms"]
y_train = train_data["labels"]

x_valid = train_data["spectrograms"]
y_valid = valid_data["labels"]

x_pretrain = finetune_data[0:len(y_train),:,:]
x_pretest = finetune_data[len(y_train):,:,:]



x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
input_shape1=(2,8,32)
 
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')

y_train = to_categorical(y_train, num_classes=num_classes)
y_valid = to_categorical(y_valid, num_classes=num_classes)


x11 = Input(shape=input_shape)
x22=Input(shape=input_shape1)

# *********************************TF_CNN**************************************************
# convert class vectors to binary class matrices
x11 = Input(shape=input_shape)
x22=Input(shape=input_shape1)
# ***************************************************Tf_capsule******************************************************
def dense_cap_block(capcnn):
    xx = capcnn
    capcnn = Capsule(16, (8, 32), 5, True)(capcnn)
    capcnn = Concatenate(axis=-1)([capcnn, xx])
    return capcnn
pretrain=Flatten()(x22)
pretrain = Dense(128, activation="relu")(pretrain)
cap_cnn = Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x11)
cap_cnn = squash(Capsule(8, (4, 32), 3, True)(cap_cnn), name_index=1)
tf_cap = Flatten()(cap_cnn)
tf_cap = Dense(128, activation="relu")(tf_cap)
# ***************************************************Tf_cnn******************************************************
xx1 = Conv2D(32, kernel_size=(8, 1), activation='relu')(x11)
xx1 = MaxPooling2D(pool_size=(2, 4))(xx1)
xx1 = Conv2D(64, kernel_size=(8, 1), activation='relu')(xx1)
xx1 = MaxPooling2D(pool_size=(2, 4))(xx1)
xx1 = Conv2D(64, kernel_size=(2, 1), activation='relu')(xx1)
xx1 = Flatten()(xx1)


# frequency_conv

xx2 = Conv2D(32, kernel_size=(1, 10), activation='relu')(x11)
xx2 = MaxPooling2D(pool_size=(2, 4))(xx2)
xx2 = Conv2D(64, kernel_size=(1, 10), activation='relu')(xx2)
xx2 = MaxPooling2D(pool_size=(2, 4))(xx2)
xx2 = Conv2D(64, kernel_size=(1, 4), activation='relu')(xx2)
xx2 = Flatten()(xx2)

# time_frequency_conv
xx3 = Conv2D(32, kernel_size=(5, 5), activation='relu')(x11)
xx3 = MaxPooling2D(pool_size=(2, 2))(xx3)
xx3 = Conv2D(64, kernel_size=(5, 5), activation='relu')(xx3)
xx3 = MaxPooling2D(pool_size=(2, 4))(xx3)
xx3 = Conv2D(64, kernel_size=(2, 2), activation='relu')(xx3)
xx3 = Flatten()(xx3)

tf_cnn = Concatenate()([xx1, xx2, xx3])
tf_cnn = Dense(1024, activation="relu")(tf_cnn)
# *********************************Gate_Fusion*********************************************
def kronecker_product(mat1, mat2):
    n1 = mat1.get_shape()[1]
    n2 = mat2.get_shape()[1]
    mat1 = RepeatVector(n2)(mat1)
    mat1 = concatenate([mat1[:, :, i] for i in range(n1)], axis=-1)
    mat2 = Flatten()(RepeatVector(n1)(mat2))
    result = multiply(inputs=[mat1, mat2])
    return result
Kronecker = Lambda(lambda tensors: kronecker_product(tensors[0], tensors[1]))
tf_cnn_size = 1024
dense_capsule_size = 128
pretrain_size = 128
is_gate = True
is_fusion = True

def gate_fusion(tf_cnn_size, dense_capsule_size, pretrain_size, is_gate, is_fusion, hidden_size=100):
    _tf_cnn = Input(shape=(tf_cnn_size,), dtype='float32', name='tf_cnn')
    _tf_cnn_out = Dense(hidden_size, activation='relu', name='tf_cnn_out')(_tf_cnn)

    _dense_capsule = Input(shape=(dense_capsule_size,), name='dense_capsule')
    _dense_capsule_out = Dense(hidden_size, activation='relu', name='dense_capsule_out')(_dense_capsule)

    _pretrain = Input(shape=(pretrain_size,), name="pretrain")
    _pretrain_out = Dense(hidden_size, activation='relu', name='pretrain_out')(_pretrain)

    if is_gate:
        _tf_cnn_gate = Dense(hidden_size, activation="sigmoid", name='tf_cnn_gate')(
            concatenate([_dense_capsule_out, _pretrain_out],
                        axis=-1))
        _dense_capsule_gate = Dense(hidden_size, activation="sigmoid", name='dense_capsule_gate')(
            concatenate([_tf_cnn_out, _pretrain_out],
                        axis=-1))
        _pretrain_gate = Dense(hidden_size, activation="sigmoid", name='pretrain_gate')(
            concatenate([_tf_cnn_out, _dense_capsule_out],
                        axis=-1))
        _tf_cnn_filtered = multiply([_tf_cnn_out, _tf_cnn_gate])
        _dense_capsule_filtered = multiply([_dense_capsule_out, _dense_capsule_gate])
        _pretrain_filtered = multiply([_pretrain_out, _pretrain_gate])

    if is_fusion:
        tfcnn_densecap_kron = Kronecker([_dense_capsule_filtered, _tf_cnn_filtered])
        tfcnn_pretrain_kron = Kronecker([_pretrain_filtered, _tf_cnn_filtered])
        densecap_pretrain_kron = Kronecker([_dense_capsule_filtered, _pretrain_filtered])
        datas = [_tf_cnn_out, _dense_capsule_out, _pretrain_out, tfcnn_densecap_kron,
                 tfcnn_pretrain_kron, densecap_pretrain_kron]
    else:
        datas = [_tf_cnn_out, _dense_capsule_out, _pretrain_out]

    cat_data = concatenate(datas)
    cat_hidden = Dropout(0.5)(cat_data)

    cat_out = Dense(1024, activation="relu")(cat_hidden)
    _model = Model(inputs=[_tf_cnn, _dense_capsule, _pretrain], outputs=cat_out)
    return _model


fusion_Model = gate_fusion(tf_cnn_size, dense_capsule_size, pretrain_size, is_gate, is_fusion)
fusion_data = fusion_Model([tf_cnn, tf_cap, pretrain])
# fusion_data = Flatten()(fusion_data)

# **********************************************************************************

intermedia_output = Dense(1024, activation="relu", name="intermedia_output")(fusion_data)

x = Dropout(0.25)(intermedia_output)
output = Dense(num_classes, activation='softmax')(x)

model = Model([x11,x22], output)
model.summary()



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_valid, y_valid))

# get feature
intermediate_layer_model = Model(inputs=model.input,
                                  outputs=model.get_layer(name='intermedia_output').output)
train_output = intermediate_layer_model.predict(x_train)
valid_output = intermediate_layer_model.predict(x_valid)


# save features
feature_path = os.path.join(os.path.split(__file__)[0], "segment_feat.pkl")

pickle.dump({
            'train_data':train_output,
            'valid_data':valid_output,
            'train_label':y_train,
            'valid_label':y_valid,
            "segment_nums_train":train_data["segment_nums"],
            "segment_nums_valid":valid_data["segment_nums"],
            "speaker_train":train_data["speaker"],
            "speaker_valid":valid_data["speaker"],
            "dialog_lengths_train":train_data["dialog"],
            "dialog_lengths_valid":valid_data["dialog"],
            },
            open(feature_path, "wb"))
