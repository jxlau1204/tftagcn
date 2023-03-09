import h5py
import scipy.io as sio
import numpy as np, pandas as pd
from collections import defaultdict
import pickle
import os
import numpy as np
model_name = "cnn"


feature_path = os.path.join(os.path.split(__file__)[0], "segment_feat.pkl")

data = pickle.load(open(feature_path, "rb"))


train_index = data["segment_nums_train"]
data_train = data["train_data"]
train_label = data["train_label"]

valid_index = data["segment_nums_valid"]
data_valid = data["valid_data"]
valid_label = data["valid_label"]



def main():
    train_seg_nums = []
    valid_seg_nums = []

    for i in range(train_index.shape[0]):
        train_seg_nums.append(train_index[i])
    for i in range(valid_index.shape[0]):
        valid_seg_nums.append(valid_index[i])

    max_len = 300
    pad = np.asarray([300 for i in range(data_train[0][:].shape[0])])

    print("Mapping train")

    train_data_X = []
    train_data_Y = []
    train_length = []
    seg_counts = 0
    count = 0
    for value in train_index:
        lst_X, lst_Y, lst_speaker = [], [], []
        ctr = 0
        for i in range(seg_counts, seg_counts + min(value, max_len)):
            ctr += 1
            lst_X.append(data_train[i])

        train_length.append(ctr)
        for i in range(ctr, max_len):
            lst_X.append(pad)

        train_data_X.append(np.asarray(lst_X))
        train_data_Y.append(train_label[seg_counts])
        seg_counts += value
        count += 1

    train_data_X = np.asarray(train_data_X)
    print("Dumping train data")
    train_data_path = os.path.split(__file__)[0] + f"/train_utterance_feat.pkl"
    with open(train_data_path, 'wb') as handle:
        pickle.dump((train_data_X, np.asarray(train_data_Y), max_len, train_length, data["speaker_train"], data["dialog_lengths_train"]), handle, protocol=4)

    valid_data_X = []
    valid_data_Y = []
    valid_length = []
    count = 0
    print("Mapping valid")

    seg_counts = 0
    for value in valid_index:
        lst_X, lst_Y, lst_speaker = [], [], []
        ctr = 0
        for i in range(seg_counts, seg_counts + min(value, max_len)):
            ctr += 1
            lst_X.append(data_valid[i])
        valid_length.append(ctr)
        for i in range(ctr, max_len):
            lst_X.append(pad)
        valid_data_X.append(np.asarray(lst_X))
        valid_data_Y.append(valid_label[seg_counts])
        seg_counts += value
        count += 1

    valid_data_X = np.asarray(valid_data_X)
    print(train_data_X.shape, valid_data_X.shape, len(train_length), len(valid_length))

    print("Dumping valid data")
    valid_data_path = os.path.split(__file__)[0] + f"/valid_utterance_feat.pkl"
    with open(valid_data_path, 'wb') as handle:
        pickle.dump((valid_data_X, np.asarray(valid_data_Y), max_len, valid_length, data["speaker_valid"], data["dialog_lengths_valid"]), handle, protocol=4)


if __name__ == "__main__":
    main()
