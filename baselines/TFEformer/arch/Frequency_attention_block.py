import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.fft as fft


class freq_att(nn.Module):
    def __init__(self, dim_input,num_head,dropout):
        super(freq_att, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)
        self.laynorm = nn.LayerNorm([dim_input])

    def forward(self, x):
        q = fft.fftn(self.dropout(self.query(x)))
        k = fft.fftn(self.dropout(self.key(x)))
        k = k.transpose(-2, -1)
        v = fft.fftn(self.dropout(self.value(x)))

        ### real
        q_real = q.real
        k_real = k.real
        v_real = v.real

        ### imag
        q_imag = q.imag
        k_imag = k.imag
        v_imag = v.imag

        result_real = 0.0
        result_imag = 0.0
        for i in range(self.num_head):
            line_real = self.softmax(q_real @ k_real) @ v_real
            line_real = line_real.unsqueeze(-1)

            line_imag = self.softmax(q_imag @ k_imag) @ v_imag
            line_imag = line_imag.unsqueeze(-1)

            if i < 1:
                result_real = line_real
                result_imag = line_imag
            else:
                result_real = torch.cat([result_real,line_real],dim=-1)
                result_imag = torch.cat([result_imag, line_imag], dim=-1)
        result = torch.sqrt(result_real*result_real + result_imag*result_imag)
        result = self.dropout(self.linear1(result).squeeze(-1)) + x
        result = self.laynorm(result)
        return result