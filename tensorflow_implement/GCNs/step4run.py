import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from GCNsModel import *
from scipy.io import loadmat
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support

class MyDataset(Dataset):
    def __init__(self,
                 split,
                 data_path= "tensorflow_implement/CNN/LSTM/utterance_feat.pkl", 
                 **kward) -> None:
        super().__init__()
        data = pickle.load(open(data_path, "rb"))
        
        self.feats = data[f"{split}_data"]
        self.labels = data[f"{split}_label"].argmax(-1)
        self.speakers = data[f"speaker_{split}"]
        self.dialog_lengths = data[f"dialog_lengths_{split}"]
        self.feats_dialog = torch.split(torch.FloatTensor(self.feats), list(self.dialog_lengths))
        self.labels_dialog = torch.split(torch.LongTensor(self.labels), list(self.dialog_lengths))
        self.speakers_dialog = torch.split(torch.LongTensor(self.speakers), list(self.dialog_lengths))
    def __getitem__(self, index):
        return self.feats_dialog[index],\
            self.labels_dialog[index],\
            self.dialog_lengths[index],\
            self.speakers_dialog[index]
                
    def __len__(self):
        return len(self.dialog_lengths)
    def collate_fn(self, data):
        feats = []
        labels = []
        speakers = []
        dialog_lengths = []
        for data_i in data:
            feats.append(data_i[0])
            labels.append(data_i[1])
            dialog_lengths.append(data_i[2])
            speakers.append(data_i[3])
        return {
            "input":torch.cat(feats, axis=0),
            "speakers":torch.cat(speakers),
            "dialog_lengths":torch.LongTensor(dialog_lengths),
            "labels":torch.cat(labels)
        }
        
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DIAGCN(input_size=1024,
               hidden_size= 256,
               n_classes=4,
               device = device
               )

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
model.train()
max_val_acc = 0
min_val_loss = 10000
max_val_acc_model = 0

cur_step = 0

train_dataset = MyDataset("train")
train_data_loader = DataLoader( train_dataset,
            collate_fn=train_dataset.collate_fn,
            batch_size=10,
            num_workers=1)

valid_dataset = MyDataset("valid")

model.train()
for epoch in range(30):
    # max_val_acc = 0

    optimizer.zero_grad()
    for batch_train in train_data_loader:
    # out = model(data)
        out, train_loss = model(**batch_train)
        train_loss.backward()
        optimizer.step()
        if epoch%1==0:
            with torch.no_grad():
                out, valid_loss = model(torch.FloatTensor(valid_dataset.feats),
                                  torch.LongTensor(valid_dataset.dialog_lengths), 
                                  torch.LongTensor(valid_dataset.speakers), 
                                  torch.LongTensor(valid_dataset.labels))
                print("epoch %i: train loss: %.4f; valid loss: %.4f; valid accuracy: %.4f; valid f1-score: %.4f"
                      %(epoch, train_loss.data, valid_loss.data, 
                        accuracy_score(valid_dataset.labels, out.argmax(-1).cpu().data.numpy()),
                        f1_score(valid_dataset.labels, out.argmax(-1).cpu().data.numpy(), average ="weighted")
                        )
                      )
                