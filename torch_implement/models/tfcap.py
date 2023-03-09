import torch 
import torch.nn as nn  
import torch.nn.functional as F
from torch.autograd import Variable
def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sequential(
            nn.LayerNorm(in_planes),
            nn.Sigmoid()
        )

    def forward(self, x): # x 的输入格式是：[batch_size, C, H]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out.squeeze() + max_out.squeeze()
        return self.sigmoid(out).unsqueeze(-1)

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, in_channels, kernel_size=2, stride=1,
                 num_iterations= 3, dim_capsules=[4, 8]):
        super().__init__()
        self.num_iterations = num_iterations
        self.dim_capsules = dim_capsules
        self.num_capsules = num_capsules
        self.first_conv = nn.Conv2d(in_channels, num_capsules * dim_capsules[0] * dim_capsules[1], kernel_size, stride)
        self.channel_attention = ChannelAttention(num_capsules, 2)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm)
    
    def squashWithChannelAttention(self, tensor, dim=-1):
        tensor = tensor.reshape(*tensor.shape[:2], -1)
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm) + 1e-6)

    def forward(self, x):
        batch_size = x.shape[0]
        u_hat_vecs = self.first_conv(x)

        u_hat_vecs = u_hat_vecs.reshape(batch_size, self.num_capsules, self.dim_capsules[0], self.dim_capsules[1], -1)

        u_hat_vecs = u_hat_vecs.permute(0, 1, 4, 2, 3)        

        b = torch.zeros_like(u_hat_vecs[:,:,:,0,0]).to(u_hat_vecs.device)
        
        for i in range(self.num_iterations):
            c = softmax(b, 1)
            o = torch.einsum('bin,binjk->bijk', c, u_hat_vecs)
            o_temp = o.reshape(o.shape[0], o.shape[1], -1)
            if i != self.num_iterations - 1:
                o_temp = F.normalize(o_temp, dim=-1)
                o = o_temp.reshape(*o.shape)
                
                b = torch.einsum('bijk,binjk->bin', o, u_hat_vecs)
                
        return self.squashWithChannelAttention(o)
        # return self.squash(o)
    
class TFCAPLayer(nn.Module):
    def __init__(self, 
                 in_channel, 
                 hidden_channel,
                 image_size,
                 out_capsules_dims,
                 out_capsules_num,
                 dropout,
                 num_iterations = 3,
                 out_size = 128) -> None:
        super().__init__()
        self.cnns = nn.Sequential(
            nn.LayerNorm([in_channel, *image_size]),
            nn.Conv2d(in_channel, hidden_channel[0], (5, 5), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(hidden_channel[0], hidden_channel[1], (7, 7), (2, 2)),
            nn.ReLU()
        )
        with torch.no_grad():
            test_input = torch.rand(1, in_channel, *image_size)
            out_dims = self.cnns(test_input).shape
        self.capsules = nn.Sequential(
                CapsuleLayer(out_capsules_num, 
                            hidden_channel[1], 
                            kernel_size=2, 
                            stride=1,
                            dim_capsules = out_capsules_dims,
                            num_iterations= num_iterations)
        )
        self.output_layer = nn.Sequential(
            # nn.Linear(out_capsules_num * out_capsules_dims[0] * out_capsules_dims[1], out_size),
            nn.Linear(out_dims[-1] *out_dims[-2] * hidden_channel[1], out_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, input):
        log = {}
        input = input.unsqueeze(1)
        x = self.cnns(input)
        # x = self.capsules(x)
        x = x.reshape(input.shape[0],-1)
        out = self.output_layer(x)
        return out, log