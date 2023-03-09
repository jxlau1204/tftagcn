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
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.initializers import glorot_normal

from keras.callbacks import ModelCheckpoint
import h5py
from keras.utils import to_categorical

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

x_train = train_data["spectrograms"]
y_train = train_data["labels"]

x_valid = train_data["spectrograms"]
y_valid = valid_data["labels"]


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_valid = x_valid.reshape(x_valid.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
 
x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'valid samples')

y_train = to_categorical(y_train, num_classes=num_classes)
y_valid = to_categorical(y_valid, num_classes=num_classes)
# *****************************************************************************************
# *********************************TF_CNN**************************************************
# convert class vectors to binary class matrices
x11 = Input(shape=input_shape)

# time_conv
xx1 = Conv2D(32, kernel_size=(1, 5), activation='relu')(x11)
xx1 = MaxPooling2D(pool_size=(3, 2))(xx1)
xx1 = Conv2D(64, kernel_size=(1, 5), activation='relu')(xx1)
xx1 = MaxPooling2D(pool_size=(3, 2))(xx1)
#xx1 = Conv2D(64, kernel_size=(1, 7), activation='relu')(xx1)
xx1 = Flatten()(xx1)

# frequency_conv
xx2 = Conv2D(32, kernel_size=(5, 1), activation='relu')(x11)
xx2 = MaxPooling2D(pool_size=(3, 2))(xx2)
xx2 = Conv2D(64, kernel_size=(5, 1), activation='relu')(xx2)
xx2 = MaxPooling2D(pool_size=(3, 2))(xx2)
#xx2 = Conv2D(64, kernel_size=(10, 2), activation='relu')(xx2)
xx2 = Flatten()(xx2)

# time_frequency_conv
xx3 = Conv2D(32, kernel_size=(5, 5), activation='relu')(x11)
xx3 = MaxPooling2D(pool_size=(2, 2))(xx3)
xx3 = Conv2D(64, kernel_size=(5, 5), activation='relu')(xx3)
xx3 = MaxPooling2D(pool_size=(2, 2))(xx3)
xx3 = Flatten()(xx3)

tf_cnn = Concatenate()([xx1, xx2, xx3])


intermedia_output = Dense(1024, activation="relu", name="intermedia_output")(tf_cnn)

x = Dropout(0.25)(intermedia_output)
output = Dense(num_classes, activation='softmax')(x)

model = Model(x11, output)
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
train_output = intermediate_layer_model.predict(x_train, verbose=0)
valid_output = intermediate_layer_model.predict(x_valid, verbose=0)


# save features
feature_path = os.path.join(os.path.split(__file__)[0], "segment_feat.pkl")

pickle.dump({
            'train_data':train_output,
            'valid_data':valid_output,
            'train_label':y_train,
            'valid_label':y_valid,
            "segment_nums_train":train_data["segment_nums"],
            "segment_nums_valid":valid_data["segment_nums"],
            "speaker_train":train_data["speakers"],
            "speaker_valid":valid_data["speakers"],
            "dialog_lengths_train":train_data["dialog_utterances"],
            "dialog_lengths_valid":valid_data["dialog_utterances"],
            },
            open(feature_path, "wb"))