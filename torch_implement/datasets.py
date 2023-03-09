import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch
import pandas as pd
from units import *
from torch.utils.data import DataLoader
import h5py
from scipy.io import loadmat

class IEMOCAP(Dataset):
    def __init__(self, 
                 split, 
                 train_path= "data/iemocap_vad/spectrograms/cross_flod1/train.pkl", 
                 valid_path = "data/iemocap_vad/spectrograms/cross_flod1/valid.pkl", 
                 batch_time = 100,
                 **kward) -> None:
        super().__init__()
        self.batch_time = batch_time
        if split == "train":
            data = pickle.load(open(train_path, "rb"))
        else:
            data = pickle.load(open(valid_path, "rb"))
            
        self.spectrograms = torch.tensor(data["spectrograms"], dtype=torch.float32)
        self.segment_nums = torch.tensor(data["segment_nums"])
        self.spectrograms = torch.split(self.spectrograms, self.segment_nums.tolist())
        self.labels = list(data["utterances_label"])
        self.len = len(self.labels)
        self.sizes = self.segment_nums * 0.025
        print(": %d"%self.len)
    def __getitem__(self, index):
        return torch.FloatTensor(self.spectrograms[index]),\
               torch.LongTensor([self.segment_nums[index]]),\
               torch.LongTensor([self.labels[index]])
    def __len__(self):
        return self.len
    def batch_sampler(self):
        assert self.sizes.max() <= self.batch_time
        return MyBatchSampler(self.batch_by_size(self.sizes.argsort(), batch_time=self.batch_time))
    def batch_by_size(self, indices, batch_time=None):
        batches = []
        batch_i = []
        acc_sum = 0
        for index_i in indices:
            if acc_sum+self.sizes[index_i] > batch_time:
                batches.append(np.array(batch_i))
                acc_sum = self.sizes[index_i]
                batch_i = [index_i]
            else:
                acc_sum += self.sizes[index_i]
                batch_i.append(index_i)
        batches.append(np.array(batch_i))
        return batches
    def collate_fn(self, data):
        spectrograms = []
        lengths = []
        labels = []
        for data_i in data:
            spectrograms.append(data_i[0])
            lengths.append(data_i[1])
            labels.append(data_i[2])
        return {
            "spectrograms":torch.cat(spectrograms, axis=0),
            "lengths": torch.cat(lengths),
            "labels":torch.cat(labels)
        }
class IEMOCAPDIA(Dataset):
    def __init__(self, 
                 split, 
                 train_path= "data/iemocap_vad/spectrograms/cross_flod1/train.pkl", 
                 valid_path = "data/iemocap_vad/spectrograms/cross_flod1/valid.pkl", 
                 pretrain_path = None,
                 load_pretrain = False,
                 batch_time = 200,
                 **kward) -> None:
        super().__init__()
        self.batch_time = batch_time
        self.load_pretrain = load_pretrain
        if pretrain_path and load_pretrain:
            self.pretrain = torch.FloatTensor(pickle.load(open(pretrain_path, "rb"))["embeddings"])
        else:
            self.load_pretrain = False

        if split == "train":
            data = pickle.load(open(train_path, "rb"))
        else:
            data = pickle.load(open(valid_path, "rb"))
            
        self.spectrograms = torch.tensor(data["spectrograms"], dtype=torch.float32)
        self.segment_nums = torch.tensor(data["segment_nums"])
        

        self.dialog_lengths = torch.tensor(data["dialog_utterances"])
        self.dialogs_segment_nums = torch.split(self.segment_nums, self.dialog_lengths.tolist())
        self.dialogs_segment_nums = np.array([int(x.sum()) for x in self.dialogs_segment_nums])
        
        self.spectrograms_dialog = torch.split(self.spectrograms, self.dialogs_segment_nums.tolist())
        if self.load_pretrain:
            self.pretrain_dialog = torch.split(self.pretrain, self.dialogs_segment_nums.tolist())
        self.segment_nums_dialog = torch.split(self.segment_nums, self.dialog_lengths.tolist())
        self.labels = torch.split(torch.tensor(data["utterances_label"]), self.dialog_lengths.tolist())
        self.speakers = torch.split(torch.tensor(data["speakers"]), self.dialog_lengths.tolist())
        self.len = len(self.dialog_lengths)
        self.sizes = self.dialogs_segment_nums * 0.025
        print(": %d"%self.len)
        print("max size: %f"%self.sizes.max())
    def __getitem__(self, index):
        if self.load_pretrain:
            return self.spectrograms_dialog[index],\
                self.segment_nums_dialog[index],\
                self.labels[index],\
                self.dialog_lengths[index],\
                self.pretrain_dialog[index],\
                self.speakers[index]
        else:
            return self.spectrograms_dialog[index],\
                self.segment_nums_dialog[index],\
                self.labels[index],\
                self.dialog_lengths[index],\
                self.speakers[index]
                
    def __len__(self):
        return self.len
    def batch_sampler(self):
        assert self.sizes.max() <= self.batch_time
        return MyBatchSampler(self.batch_by_size(self.sizes.argsort(), batch_time=self.batch_time))
    def batch_by_size(self, indices, batch_time=None):
        batches = []
        batch_i = []
        acc_sum = 0
        for index_i in indices:
            if acc_sum+self.sizes[index_i] > batch_time:
                batches.append(np.array(batch_i))
                acc_sum = self.sizes[index_i]
                batch_i = [index_i]
            else:
                acc_sum += self.sizes[index_i]
                batch_i.append(index_i)
        batches.append(np.array(batch_i))
        return batches
    def collate_fn(self, data):
        spectrograms = []
        lengths = []
        labels = []
        speakers = []
        dialog_lengths = []
        pretrain = []
        for data_i in data:
            spectrograms.append(data_i[0])
            lengths.append(data_i[1])
            labels.append(data_i[2])
            dialog_lengths.append(data_i[3])
            if self.load_pretrain:
                pretrain.append(data_i[4])
            speakers.append(data_i[-1])
        if self.load_pretrain:
            return {
                "spectrograms":torch.cat(spectrograms, axis=0),
                "lengths": torch.cat(lengths),
                "labels":torch.cat(labels),
                "speakers":torch.cat(speakers),
                "dialog_lengths":torch.LongTensor(dialog_lengths),
                "pretrain_embedding":torch.cat(pretrain, axis=0),
            }
        else:
            return {
                "spectrograms":torch.cat(spectrograms, axis=0),
                "lengths": torch.cat(lengths),
                "labels":torch.cat(labels),
                "speakers":torch.cat(speakers),
                "dialog_lengths":torch.LongTensor(dialog_lengths)
            }
class MELD(Dataset):
    def __init__(self, 
                 split, 
                 train_path= "data/meld4/spectrograms/train.pkl", 
                 valid_path = "data/meld4/spectrograms/valid.pkl", 
                 test_path = "data/meld4/spectrograms/test.pkl", 
                 batch_time = 100,
                 **kward) -> None:
        super().__init__()
        self.batch_time = batch_time
        if split == "train":
            data = pickle.load(open(train_path, "rb"))
        elif split == "valid":
            data = pickle.load(open(valid_path, "rb"))
        else:
            data = pickle.load(open(test_path, "rb"))
            
        self.spectrograms = torch.tensor(data["spectrograms"], dtype=torch.float32)
        self.segment_nums = torch.tensor(data["segment_nums"])
        self.spectrograms = torch.split(self.spectrograms, self.segment_nums.tolist())
        self.labels = list(data["utterances_label"])
        self.len = len(self.labels)
        self.sizes = self.segment_nums * 0.025
        print(": %d"%self.len)
    def __getitem__(self, index):
        return torch.FloatTensor(self.spectrograms[index]),\
               torch.LongTensor([self.segment_nums[index]]),\
               torch.LongTensor([self.labels[index]])
    def __len__(self):
        return self.len
    def batch_sampler(self):
        assert self.sizes.max() <= self.batch_time
        return MyBatchSampler(self.batch_by_size(self.sizes.argsort(), batch_time=self.batch_time))
    def batch_by_size(self, indices, batch_time=None):
        batches = []
        batch_i = []
        acc_sum = 0
        for index_i in indices:
            if acc_sum+self.sizes[index_i] > batch_time:
                batches.append(np.array(batch_i))
                acc_sum = self.sizes[index_i]
                batch_i = [index_i]
            else:
                acc_sum += self.sizes[index_i]
                batch_i.append(index_i)
        batches.append(np.array(batch_i))
        return batches
    def collate_fn(self, data):
        spectrograms = []
        lengths = []
        labels = []
        for data_i in data:
            spectrograms.append(data_i[0])
            lengths.append(data_i[1])
            labels.append(data_i[2])
        return {
            "spectrograms":torch.cat(spectrograms, axis=0),
            "lengths": torch.cat(lengths),
            "labels":torch.cat(labels)
        }
class MELDDIA(Dataset):
    def __init__(self, 
                 split, 
                 train_path= "data/meld6/spectrograms/train.pkl", 
                 valid_path = "data/meld6/spectrograms/valid.pkl", 
                 test_path = "data/meld6/spectrograms/test.pkl", 
                 pretrain_path = None,
                 load_pretrain = False,
                 batch_time = 200,
                 **kward) -> None:
        super().__init__()
        self.batch_time = batch_time
        self.load_pretrain = load_pretrain
        if pretrain_path and load_pretrain:
            self.pretrain = torch.FloatTensor(pickle.load(open(pretrain_path, "rb"))["embeddings"])
        else:
            self.load_pretrain = False
        self.batch_time = batch_time
        if split == "train":
            data = pickle.load(open(train_path, "rb"))
        elif split == "valid":
            data = pickle.load(open(valid_path, "rb"))
        else:
            data = pickle.load(open(test_path, "rb"))
            
        self.spectrograms = torch.tensor(data["spectrograms"], dtype=torch.float32)
        self.segment_nums = torch.tensor(data["segment_nums"])
        

        self.dialog_lengths = torch.tensor(data["dialog_utterances"])
        self.dialogs_segment_nums = torch.split(self.segment_nums, self.dialog_lengths.tolist())
        self.dialogs_segment_nums = np.array([int(x.sum()) for x in self.dialogs_segment_nums])
        
        self.spectrograms_dialog = torch.split(self.spectrograms, self.dialogs_segment_nums.tolist())
        if self.load_pretrain:
            self.pretrain_dialog = torch.split(self.pretrain, self.dialogs_segment_nums.tolist())
        self.segment_nums_dialog = torch.split(self.segment_nums, self.dialog_lengths.tolist())
        self.labels = torch.split(torch.tensor(data["utterances_label"]), self.dialog_lengths.tolist())
        self.speakers = torch.split(torch.tensor(data["speakers"]), self.dialog_lengths.tolist())
        self.len = len(self.dialog_lengths)
        self.sizes = self.dialogs_segment_nums * 0.025
        print(": %d"%self.len)
        print("max size: %f"%self.sizes.max())
    def __getitem__(self, index):
        if self.load_pretrain:
            return self.spectrograms_dialog[index],\
                self.segment_nums_dialog[index],\
                self.labels[index],\
                self.dialog_lengths[index],\
                self.pretrain_dialog[index],\
                self.speakers[index]
        else:
            return self.spectrograms_dialog[index],\
                self.segment_nums_dialog[index],\
                self.labels[index],\
                self.dialog_lengths[index],\
                self.speakers[index]
                
    def __len__(self):
        return self.len
    def batch_sampler(self):
        assert self.sizes.max() <= self.batch_time
        return MyBatchSampler(self.batch_by_size(self.sizes.argsort(), batch_time=self.batch_time))
    def batch_by_size(self, indices, batch_time=None):
        batches = []
        batch_i = []
        acc_sum = 0
        for index_i in indices:
            if acc_sum+self.sizes[index_i] > batch_time:
                batches.append(np.array(batch_i))
                acc_sum = self.sizes[index_i]
                batch_i = [index_i]
            else:
                acc_sum += self.sizes[index_i]
                batch_i.append(index_i)
        batches.append(np.array(batch_i))
        return batches
    def collate_fn(self, data):
        spectrograms = []
        lengths = []
        labels = []
        speakers = []
        dialog_lengths = []
        pretrain = []
        for data_i in data:
            spectrograms.append(data_i[0])
            lengths.append(data_i[1])
            labels.append(data_i[2])
            dialog_lengths.append(data_i[3])
            if self.load_pretrain:
                pretrain.append(data_i[4])
            speakers.append(data_i[-1])
        if self.load_pretrain:
            return {
                "spectrograms":torch.cat(spectrograms, axis=0),
                "lengths": torch.cat(lengths),
                "labels":torch.cat(labels),
                "speakers":torch.cat(speakers),
                "dialog_lengths":torch.LongTensor(dialog_lengths),
                "pretrain_embedding":torch.cat(pretrain, axis=0),
            }
        else:
            return {
                "spectrograms":torch.cat(spectrograms, axis=0),
                "lengths": torch.cat(lengths),
                "labels":torch.cat(labels),
                "speakers":torch.cat(speakers),
                "dialog_lengths":torch.LongTensor(dialog_lengths)
            }

if __name__ == "__main__":

    # ex_dataset = MELDDIA("test")
    ex_dataset = IEMOCAP("test")
    print(ex_dataset[0])
    data_loader = DataLoader( ex_dataset,
                collate_fn=ex_dataset.collate_fn,
                batch_sampler = ex_dataset.batch_sampler() if hasattr(ex_dataset, "batch_sampler") else None,
                num_workers=1)
    for batch_input in data_loader:
        print(batch_input)