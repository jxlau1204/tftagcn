import torch 
import torch.nn as nn  
from models.tcn import TemporalConvNet

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional=True, dropout = 0.5, **kwargs) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, bidirectional=bidirectional, dropout = dropout, batch_first=True)
    def forward(self, input):
        return self.lstm(input)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels,adaptive_size, kernel_size=2, dropout=0.2, attention=False) -> None:
        super().__init__()
        self.tcn = TemporalConvNet( num_inputs, num_channels, kernel_size, dropout, attention)
        self.adaptivePooling = nn.AdaptiveMaxPool1d(adaptive_size)

    def forward(self, input):
        return self.adaptivePooling(self.tcn(input))