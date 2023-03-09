import numpy as np
import random
import tensorflow as tf
import os

def seed_tensorflow(seed=1333337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

seed_tensorflow()
import keras
import pickle
from keras.models import Model
from keras.models import *
from keras.callbacks import *
from keras.layers import Dense, Dropout, Flatten, Permute, Multiply, Lambda, Add, MaxPooling1D, Conv1D, Masking
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN, LSTM, GRU, Reshape, BatchNormalization, Input, concatenate
from keras.layers import Bidirectional
from tcn_core import TCN
from tensorflow.keras.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from Capsule_Keras import *
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

print(keras.__version__)

TIME_STEPS = 300  # same as the height of the image
INPUT_SIZE = 1024
BATCH_SIZE = 128
epochs = 20
num_layers = 2
num_classes = 4

with open(os.path.split(__file__)[0] + "/../train_utterance_feat.pkl", 'rb') as handle:
    (train_data, train_label, maxlen, train_length, speaker_train, dialog_lengths_train) = pickle.load(handle)

with open(os.path.split(__file__)[0] + "/../valid_utterance_feat.pkl", 'rb') as handle:
    (valid_data, valid_label, maxlen, valid_length, speaker_valid, dialog_lengths_valid) = pickle.load(handle)

# load audio
x_train = train_data
y_train = train_label
x_valid = valid_data
y_valid = valid_label

y_train = to_categorical(y_train, num_classes)
y_valid = to_categorical(y_valid, num_classes)

# model make
input_shape_audio = (TIME_STEPS, INPUT_SIZE)

x11 = Input(shape=input_shape_audio)
# audio

x = Masking(mask_value=300)(x11)


x = Lambda(lambda y: y, output_shape=lambda s: s)(x)
x = TCN(return_sequences=True,
                         nb_filters=128,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         use_skip_connections=True)(x)

x = TCN(return_sequences=False,
                         nb_filters=256,
                         kernel_size=6,
                         dilations=[2 ** i for i in range(9)],
                         nb_stacks=1,
                         use_skip_connections=True)(x)


#x = Bidirectional(LSTM(1024, return_sequences=True))(x)
#x = Bidirectional(LSTM(1024, return_sequences=False))(x)
# x=Bidirectional(LSTM(256, return_sequences=True))(x)
# x=Capsule(6, 128, 3, True)(x11)

out = Dense(1024, activation='relu', name='getlayer')(x)
output = Dense(num_classes, activation='softmax')(out)

model = Model(x11, output)
model.summary()

filepath = os.path.split(__file__) + "/best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1)
callbacks_list = [checkpoint]

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=epochs,
          verbose=2,
          validation_data=(x_valid, y_valid), 
          callbacks=callbacks_list)

score = model.evaluate(x_valid, y_valid, verbose=0)
print('valid loss:', score[0])
print('valid accuracy:', score[1])

# save features


intermediate_layer_model = Model(inputs=model.input,
                                  outputs=model.get_layer(name='getlayer').output)

train_output = intermediate_layer_model.predict(x_train)
valid_output = intermediate_layer_model.predict(x_valid)

feature_path = os.path.join(os.path.split(__file__)[0], "utterance_feat.pkl")

pickle.dump({
            'train_data':train_output,
            'valid_data':valid_output,
            'train_label':y_train,
            'valid_label':y_valid,
            "speaker_train":speaker_train,
            "speaker_valid":speaker_valid,
            "dialog_lengths_train":dialog_lengths_train,
            "dialog_lengths_valid":dialog_lengths_valid,
            },
            open(feature_path, "wb"))