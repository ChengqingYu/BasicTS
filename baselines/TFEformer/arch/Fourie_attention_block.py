import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.fft as fft


class four_att(nn.Module):
    def __init__(self, dim_input,num_head,dropout):
        super(four_att, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)
        self.laynorm = nn.LayerNorm([dim_input])

    def forward(self, x):
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = fft.fftn(self.dropout(self.value(x)))
        result = 0.0
        for i in range(self.num_head):
            line = fft.fftn(self.softmax(q @ k)) @ v
            line = line.unsqueeze(-1)
            if i < 1:
                result = line
            else:
                result = torch.cat([result,line],dim=-1)
        result = fft.ifftn(result)
        result = result.real * result.imag
        result = self.dropout(self.linear1(result).squeeze(-1)) + x
        result = self.laynorm(result)
        return result