from models.tfcnn import TFCNNLayer, CNNLayer
from models.tfcap import TFCAPLayer
from models.gate_fusion import GataFusion
from models.diagcn import DIAGCN
from models.rnns import LSTM, TCN
import torch 
import torch.nn as nn  
import torch.nn.functional as F
import torchvision
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import datasets
from torch.nn.utils.rnn import  pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence

class MyModel(nn.Module):
    def loss(self, classes, labels, mode="classification"):
        if mode == "classification":
            if self.hyp_params.get("used_focal_loss", False):
                return torchvision.ops.sigmoid_focal_loss(classes, F.one_hot(labels, len(self.hyp_params.classes_name)).float().to(labels.device), reduction="mean")
            return F.cross_entropy(classes, labels)
        else:
            return F.mse_loss(classes.view(-1), labels)        
    def metric(self, out, labels):
        out = out.data.cpu().numpy()
        pred_out = out.argmax(-1)
        labels = labels.cpu().numpy()
        accs = []
        nums = []
        classes = self.hyp_params.classes_name
        n_classes = len(classes)
        single_accuracy = ""
        for i in range(n_classes):
            index_i = np.argwhere(labels == i)
            if len(index_i)==0:
                accs.append(0)
                nums.append(0)
                continue
            else:
                accs.append(accuracy_score(labels[index_i], pred_out[index_i]))
                nums.append(len(index_i))
            single_accuracy = single_accuracy + "%s:%.4f,"%(classes[i], accs[-1])
        unweight_accuracy = sum(accs)/n_classes

        accuracy = accuracy_score(labels, pred_out)
        f1_micro = f1_score(labels, pred_out, average="micro")
        f1_macro = f1_score(labels, pred_out, average="macro")
        f1_weighted = f1_score(labels, pred_out, average="weighted")
        # c_m = confusion_matrix(labels, pred_out)
        res = {
            "unweight_accuracy":unweight_accuracy, 
            "single_accuracy":single_accuracy,
            "accuracy":accuracy, 
            "f1_micro":f1_micro, 
            "f1_macro":f1_macro, 
            "f1_weighted":f1_weighted
            # "c_m":torch.Tensor(c_m),
        }
        return res

class CNN(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.cnn = CNNLayer(image_size=hyparams.image_size, **hyparams.cnn)
        self.res = nn.Linear(hyparams.cnn.out_size, len(hyparams.classes_name))
    def forward(self, spectrograms, **kwargs):
        log = {}
        cnn_feats, res = self.cnn(spectrograms)
        out = self.res(cnn_feats)
        return out, log

class CNN_LSTM(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        # self.layer_norm = nn.LayerNorm()
        self.cnn = CNNLayer(image_size=hyparams.image_size, **hyparams.cnn)
        self.lstm = LSTM(input_size=hyparams.cnn.out_size, **hyparams.lstm)
        self.classify = nn.Sequential(
            nn.Linear(hyparams.lstm.hidden_size * (2 if hyparams.lstm.bidirectional else 1), hyparams.classify.hidden_size),
            nn.Dropout(hyparams.classify.dropout),
            nn.ReLU(),
            nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
        )
        self.res = nn.Linear(hyparams.cnn.out_size, hyparams.lstm.hidden_size * (2 if hyparams.lstm.bidirectional else 1))
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, res = self.cnn(spectrograms)
        log.update(res)
        squences = torch.split(cnn_feats, lengths.tolist())
        padded_sequence = pad_sequence(squences, True)
        h_n, c_n = self.lstm(padded_sequence)
        rnn_out = h_n[list(range(h_n.shape[0])), lengths-1, :]
        rnn_out = rnn_out + self.res(padded_sequence.sum(-2))/lengths.view(-1,1)
        out = self.classify(rnn_out)
        return out, log
    
class CNN_TCN(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        # self.layer_norm = nn.LayerNorm()
        self.cnn = CNNLayer(image_size=hyparams.image_size, **hyparams.cnn)
        self.tcn = TCN(num_inputs=hyparams.cnn.out_size, **hyparams.tcn)
        self.classify = nn.Sequential(
            nn.Linear(hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size, hyparams.classify.hidden_size),
            nn.Dropout(hyparams.classify.dropout),
            nn.ReLU(),
            nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
        )
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, res = self.cnn(spectrograms)
        log.update(res)
        squences = torch.split(cnn_feats, lengths.tolist())
        padded_squences = pad_sequence(squences, True)
        rnn_out = self.tcn(padded_squences.permute(0,2,1))
        out = self.classify(rnn_out.view(rnn_out.shape[0],-1))
        return out, log

class TFCNN(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size,**hyparams.tfcnn)
        self.res = nn.Linear(hyparams.tfcnn.out_size, len(hyparams.classes_name))
    def forward(self, spectrograms, **kwargs):
        log = {}
        cnn_feats, res = self.tfcnn(spectrograms)
        out = self.res(cnn_feats)
        return out, log

class TFCNN_LSTM(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size, **hyparams.tfcnn)
        self.lstm = LSTM(input_size=hyparams.tfcnn.out_size, **hyparams.lstm)
        self.classify = nn.Sequential(
            nn.Linear(hyparams.lstm.hidden_size * 2, hyparams.classify.hidden_size),
            nn.Dropout(hyparams.classify.dropout),
            nn.ReLU(),
            nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
        )
        self.res = nn.Linear(hyparams.tfcnn.out_size, hyparams.lstm.hidden_size * 2)
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, res = self.tfcnn(spectrograms)
        log.update(res)
        squences = torch.split(cnn_feats, lengths.tolist())
        padded_sequence = pad_sequence(squences, True)
        h_n, c_n = self.lstm(padded_sequence)
        rnn_out = h_n[:,-1,:]
        rnn_out = rnn_out + self.res(padded_sequence.sum(-2))/lengths.view(-1,1)
        out = self.classify(rnn_out)
        return out, log

class TFCNN_TCN(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size, **hyparams.tfcnn)
        self.tcn = TCN(num_inputs=hyparams.tfcnn.out_size, **hyparams.tcn)
        self.classify = nn.Sequential(
            nn.Linear(hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size, hyparams.classify.hidden_size),
            nn.Dropout(hyparams.classify.dropout),
            nn.ReLU(),
            nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
        )
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, res = self.tfcnn(spectrograms)
        log.update(res)
        squences = torch.split(cnn_feats, lengths.tolist())
        padded_squences = pad_sequence(squences, True)
        rnn_out = self.tcn(padded_squences.permute(0,2,1))
        out = self.classify(rnn_out.view(rnn_out.shape[0],-1))
        # pack
        return out, log

class TFCAP(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size,**hyparams.tfcap)
        self.res = nn.Linear(hyparams.tfcap.out_size, len(hyparams.classes_name))
    def forward(self, spectrograms, **kwargs):
        log = {}
        cnn_feats, res = self.tfcap(spectrograms)
        out = self.res(cnn_feats)
        return out, log

class TFCAP_LSTM(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size, **hyparams.tfcap)
        self.lstm = LSTM(input_size=hyparams.tfcap.out_size, **hyparams.lstm)
        self.classify = nn.Sequential(
            nn.Linear(hyparams.lstm.hidden_size * 2, hyparams.classify.hidden_size),
            nn.Dropout(hyparams.classify.dropout),
            nn.ReLU(),
            nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
        )
        self.res = nn.Linear(hyparams.tfcap.out_size, hyparams.lstm.hidden_size * 2)
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, res = self.tfcap(spectrograms)
        log.update(res)
        squences = torch.split(cnn_feats, lengths.tolist())
        padded_sequence = pad_sequence(squences, True)
        h_n, c_n = self.lstm(padded_sequence)
        rnn_out = h_n[:,-1,:]
        rnn_out = rnn_out + self.res(padded_sequence.sum(-2))/lengths.view(-1,1)
        out = self.classify(rnn_out)
        return out, log

class TFCAP_TCN(MyModel):
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size, **hyparams.tfcap)
        self.tcn = TCN(num_inputs=hyparams.tfcap.out_size, **hyparams.tcn)
        self.classify = nn.Sequential(
            nn.Linear(hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size, hyparams.classify.hidden_size),
            nn.Dropout(hyparams.classify.dropout),
            nn.ReLU(),
            nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
        )
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, res = self.tfcap(spectrograms)
        log.update(res)
        squences = torch.split(cnn_feats, lengths.tolist())
        padded_squences = pad_sequence(squences, True)
        rnn_out = self.tcn(padded_squences.permute(0,2,1))
        out = self.classify(rnn_out.view(rnn_out.shape[0],-1))
        # pack
        return out, log

class TFCNCAP_LSTM_Layer(MyModel):
    # w TFCNN TFCap, wo VQ-ESR, wo GFfusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size,**hyparams.tfcnn)
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size,**hyparams.tfcap)
        self.lstm = LSTM(input_size=hyparams.tfcap.out_size + hyparams.tfcnn.out_size, **hyparams.lstm)

    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, _log = self.tfcnn(spectrograms)
        log.update(_log)
        cap_feats, _log = self.tfcap(spectrograms)
        log.update(_log)
        seg_feats = torch.cat([cnn_feats, cap_feats], -1)
        squences = torch.split(seg_feats, lengths.tolist())
        padded_sequence = pad_sequence(squences, True)
        h_n, c_n = self.lstm(padded_sequence)
        rnn_out = h_n[:,-1,:]
        return rnn_out.view(rnn_out.shape[0],-1), log
    
class TFCNCAP_TCN_Layer(MyModel):
    # w TFCNN TFCap, wo VQ-ESR, wo GFfusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size,**hyparams.tfcnn)
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size,**hyparams.tfcap)
        self.tcn = TCN(num_inputs=hyparams.tfcap.out_size + hyparams.tfcnn.out_size, **hyparams.tcn)

    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        cnn_feats, _log = self.tfcnn(spectrograms)
        log.update(_log)
        cap_feats, _log = self.tfcap(spectrograms)
        log.update(_log)
        seg_feats = torch.cat([cnn_feats, cap_feats], -1)
        squences = torch.split(seg_feats, lengths.tolist())
        padded_squences = pad_sequence(squences, True)
        rnn_out = self.tcn(padded_squences.permute(0,2,1))
        return rnn_out.view(rnn_out.shape[0],-1), log
    
class TFCNCAP_TCN(MyModel):
    # w TFCNN TFCap, wo VQ-ESR, wo GFfusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_layer = TFCNCAP_TCN_Layer(hyparams)
        self.classify = nn.Sequential(
                    nn.Linear(hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size, hyparams.classify.hidden_size),
                    nn.Dropout(hyparams.classify.dropout),
                    nn.ReLU(),
                    nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
                )
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        feats, _log = self.tfcncap_layer(spectrograms, lengths, **kwargs)
        log.update(_log)
        out = self.classify(feats)
        return out, log
    
class TFCNCAP_LSTM(MyModel):
    # w TFCNN TFCap, wo VQ-ESR, wo GFfusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_layer = TFCNCAP_LSTM_Layer(hyparams)
        self.classify = nn.Sequential(
                    nn.Linear(hyparams.lstm.hidden_size * 2, hyparams.classify.hidden_size),
                    nn.Dropout(hyparams.classify.dropout),
                    nn.ReLU(),
                    nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
                )
    def forward(self, spectrograms, lengths, **kwargs):
        log = {}
        feats, _log = self.tfcncap_layer(spectrograms, lengths, **kwargs)
        log.update(_log)
        out = self.classify(feats)
        return out, log

class TFCNCAP_CIRC_LSTM_Layer(MyModel):
    # w TFCNN TFCap, w VQ-ESR, wo GFusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size,**hyparams.tfcnn)
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size,**hyparams.tfcap)
        self.lstm = LSTM(input_size=hyparams.tfcap.out_size + hyparams.tfcnn.out_size + hyparams.pretrain_size, **hyparams.lstm)

    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        cnn_feats, _log = self.tfcnn(spectrograms)
        log.update(_log)
        cap_feats, _log = self.tfcap(spectrograms)
        log.update(_log)
        seg_feats = torch.cat([cnn_feats, cap_feats, pretrain_embedding], -1)
        squences = torch.split(seg_feats, lengths.tolist())
        padded_sequence = pad_sequence(squences, True)
        h_n, c_n = self.lstm(padded_sequence)
        rnn_out = h_n[:,-1,:]
        return rnn_out.view(rnn_out.shape[0],-1), log
    
class TFCNCAP_CIRC_TCN_Layer(MyModel):
    # w TFCNN TFCap, w VQ-ESR, wo GFusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size,**hyparams.tfcnn)
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size,**hyparams.tfcap)
        self.tcn = TCN(num_inputs=hyparams.tfcap.out_size + hyparams.tfcnn.out_size + hyparams.pretrain_size, **hyparams.tcn)

    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        cnn_feats, _log = self.tfcnn(spectrograms)
        log.update(_log)
        cap_feats, _log = self.tfcap(spectrograms)
        log.update(_log)
        seg_feats = torch.cat([cnn_feats, cap_feats, pretrain_embedding], -1)
        squences = torch.split(seg_feats, lengths.tolist())
        padded_squences = pad_sequence(squences, True)
        rnn_out = self.tcn(padded_squences.permute(0,2,1))
        return rnn_out.view(rnn_out.shape[0],-1), log
    
class TFCNCAP_CIRC_TCN(MyModel):
    # w TFCNN TFCap, w VQ-ESR, wo GFusion, w TACN, TCN
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_circ_layer = TFCNCAP_CIRC_TCN_Layer(hyparams)
        self.classify = nn.Sequential(
                    nn.Linear(hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size, hyparams.classify.hidden_size),
                    nn.Dropout(hyparams.classify.dropout),
                    nn.ReLU(),
                    nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
                )
    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        feats, _log = self.tfcncap_circ_layer(spectrograms, lengths, pretrain_embedding, **kwargs)
        log.update(_log)
        out = self.classify(feats)
        return out, log
    
class TFCNCAP_CIRC_BLSTM(MyModel):
    # w TFCNN TFCap, w VQ-ESR, wo GFusion, w BLSTM
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_circ_layer = TFCNCAP_CIRC_LSTM_Layer(hyparams)
        self.classify = nn.Sequential(
                    nn.Linear(hyparams.lstm.hidden_size * 2, hyparams.classify.hidden_size),
                    nn.Dropout(hyparams.classify.dropout),
                    nn.ReLU(),
                    nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
                )
    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        feats, _log = self.tfcncap_circ_layer(spectrograms, lengths, pretrain_embedding, **kwargs)
        log.update(_log)
        out = self.classify(feats)
        return out, log
    

class TFCNCAP_BUTTLE_TCN_Layer(MyModel):
    # w TFCNN TFCap, w VQ-ESR, w GFusion, w BLSTM
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size,**hyparams.tfcnn)
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size,**hyparams.tfcap)
        self.tcn = TCN(num_inputs=hyparams.tfcap.out_size + hyparams.tfcnn.out_size + hyparams.pretrain_size, **hyparams.tcn)
    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        #tf_cnn
        tf_cnn_feats, _log = self.tfcnn(spectrograms)
        log.update(_log)

        #tfcap
        tf_cap_feats, _log = self.tfcap(spectrograms)
        log.update(_log)
        
        # pretrained
        pretrained_feats = F.layer_norm(pretrain_embedding, pretrain_embedding.shape)
        
        gate_out, _log  = self.gate_fusion(tf_cnn_feats, tf_cap_feats, pretrained_feats)
        log.update(_log)
        
        # intra-utterance
        squences = torch.split(gate_out, lengths.tolist())
        padded_squences = pad_sequence(squences, True)
        rnn_out = self.tcn(padded_squences.permute(0,2,1))
        out = rnn_out.view(rnn_out.shape[0],-1)
        return out, log

class TFCNCAP_BUTTLE_LSTM_Layer(MyModel):
    # w TFCNN TFCap, w VQ-ESR, w GFusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcnn = TFCNNLayer(image_size=hyparams.image_size,**hyparams.tfcnn)
        self.tfcap = TFCAPLayer(image_size=hyparams.image_size,**hyparams.tfcap)
        self.lstm = LSTM(input_size=hyparams.tfcap.out_size + hyparams.tfcnn.out_size + hyparams.pretrain_size, **hyparams.lstm)
    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        #tf_cnn
        tf_cnn_feats, _log = self.tfcnn(spectrograms)
        log.update(_log)

        #tfcap
        tf_cap_feats, _log = self.tfcap(spectrograms)
        log.update(_log)
        
        # pretrained
        pretrained_feats = F.layer_norm(pretrain_embedding, pretrain_embedding.shape)
        
        gate_out, _log  = self.gate_fusion(tf_cnn_feats, tf_cap_feats, pretrained_feats)
        log.update(_log)
        
        # intra-utterance
        squences = torch.split(gate_out, lengths.tolist())
        padded_sequence = pad_sequence(squences, True)
        h_n, c_n = self.lstm(padded_sequence)
        rnn_out = h_n[:,-1,:]
        return rnn_out.view(rnn_out.shape[0],-1), log
    
class TFCNCAP_BUTTLE_TCN(MyModel):
    # w TFCNN TFCap, w VQ-ESR, w GFusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tdcncap_buttle_layer = TFCNCAP_BUTTLE_TCN_Layer(hyparams)
        self.classify = nn.Sequential(
                    nn.Linear(hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size, hyparams.classify.hidden_size),
                    nn.Dropout(hyparams.classify.dropout),
                    nn.ReLU(),
                    nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
                )
    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        feats, _log = self.tdcncap_buttle_layer(spectrograms, lengths, pretrain_embedding, **kwargs)
        log.update(_log)
        out = self.classify(feats)
        return out, log

class TFCNCAP_BUTTLE_LSTM(MyModel):
    # w TFCNN TFCap, w VQ-ESR, w GFusion
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_buttle_layer = TFCNCAP_BUTTLE_LSTM_Layer(hyparams)
        self.classify = nn.Sequential(
                    nn.Linear(hyparams.lstm.hidden_size * 2, hyparams.classify.hidden_size),
                    nn.Dropout(hyparams.classify.dropout),
                    nn.ReLU(),
                    nn.Linear(hyparams.classify.hidden_size, len(hyparams.classes_name))
                )
    def forward(self, spectrograms, lengths, pretrain_embedding, **kwargs):
        log = {}
        feats, _log = self.tfcncap_buttle_layer(spectrograms, lengths, pretrain_embedding, **kwargs)
        log.update(_log)
        out = self.classify(feats)
        return out, log
    
    
class TF_BLGCN(MyModel):
    # w TFCNCap , w TACN, w GCN
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap = TFCNCAP_LSTM_Layer(hyparams)
        self.diagcn = DIAGCN(input_size=hyparams.lstm.hidden_size * 2,**hyparams.diagcn, n_classes= len(hyparams.classes_name))
        
    def forward(self, spectrograms, lengths, dialog_lengths, speakers, **kwards):
        log = {}
        feats, _log = self.tfcncap(spectrograms,  lengths, **kwards)
        log.update(log)
        out = self.diagcn(feats, dialog_lengths, speakers)
        return out, log
    
class TF_BLGCN_CIRC(MyModel):
    # w TFCNCap◦ , w BLSTM, w GCN
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_buttle = TFCNCAP_CIRC_LSTM_Layer(hyparams)
        self.diagcn = DIAGCN(input_size=hyparams.lstm.hidden_size * 2, **hyparams.diagcn, n_classes= len(hyparams.classes_name))
        
    def forward(self, spectrograms, pretrain_embedding, lengths, dialog_lengths, speakers, **kwards):
        log = {}
        feats, _log = self.tfcncap_buttle(spectrograms, pretrain_embedding, lengths, **kwards)
        log.update(log)
        out = self.diagcn(feats, dialog_lengths, speakers)
        return out, log
    
class TF_BLGCN_BUTTLE(MyModel):
    # w TFCNCap• , w BLSTM, w GCN
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_buttle = TFCNCAP_BUTTLE_LSTM_Layer(hyparams)
        self.diagcn = DIAGCN(input_size=hyparams.lstm.hidden_size * 2, **hyparams.diagcn, n_classes= len(hyparams.classes_name))
        
    def forward(self, spectrograms, pretrain_embedding, lengths, dialog_lengths,speakers, **kwards):
        log = {}
        feats, _log = self.tfcncap_buttle(spectrograms, pretrain_embedding, lengths, **kwards)
        log.update(log)
        out = self.diagcn(feats, dialog_lengths, speakers)
        return out, log
    
    
class TF_TAGCN(MyModel):
    # w TFCNCap , w TACN, w GCN
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap = TFCNCAP_TCN_Layer(hyparams)
        self.diagcn = DIAGCN(input_size=hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size,  **hyparams.diagcn, n_classes= len(hyparams.classes_name))
        
    def forward(self, spectrograms, lengths, dialog_lengths, speakers, **kwards):
        log = {}
        feats, _log = self.tfcncap(spectrograms,  lengths, **kwards)
        log.update(log)
        out = self.diagcn(feats, dialog_lengths, speakers)
        return out, log
    
class TF_TAGCN_CIRC(MyModel):
    # w TFCNCap◦ , w TACN, w GCN
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_buttle = TFCNCAP_CIRC_TCN_Layer(hyparams)
        self.diagcn = DIAGCN(input_size=hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size,  **hyparams.diagcn, n_classes= len(hyparams.classes_name))
        
    def forward(self, spectrograms, pretrain_embedding, lengths, dialog_lengths, speakers, **kwards):
        log = {}
        feats, _log = self.tfcncap_buttle(spectrograms, pretrain_embedding, lengths, **kwards)
        log.update(log)
        out = self.diagcn(feats, dialog_lengths, speakers)
        return out, log
    
    
    
    
class TF_TAGCN_BUTTLE(MyModel):
    # w TFCNCap• , w TACN, w GCN
    def __init__(self, hyparams) -> None:
        super().__init__()
        self.hyp_params = hyparams
        self.tfcncap_buttle = TFCNCAP_BUTTLE_TCN_Layer(hyparams)
        self.diagcn = DIAGCN(input_size=hyparams.tcn.num_channels[-1] * hyparams.tcn.adaptive_size, **hyparams.diagcn, n_classes= len(hyparams.classes_name))
        
    def forward(self, spectrograms, pretrain_embedding, lengths, dialog_lengths, speakers, **kwards):
        log = {}
        feats, _log = self.tfcncap_buttle(spectrograms, pretrain_embedding, lengths, **kwards)
        log.update(log)
        out = self.diagcn(feats, dialog_lengths, speakers)
        return out, log