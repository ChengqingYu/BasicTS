import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F


class decomp_tre_sea(nn.Module):
    def __init__(self, dim_input,num_hid,T):
        super(decomp_tre_sea, self).__init__()
        self.input_len = dim_input
        self.avepool = nn.AvgPool1d(3,padding=1,stride=1)
        self.softmax = nn.Softmax(dim=-1)
        self.embid = nn.Linear(1,num_hid)
        self.T = T
    def forward(self, x, device):

        # trend
        x_tre = self.softmax(decomp_tre_sea.trend_model(x, self.T, device)) * self.avepool(x)

        # season
        x_sea = decomp_tre_sea.seasonality_model(x,self.T,device)

        # embding
        x = self.embid(x.unsqueeze(-1)).transpose(-2, -1)
        x_tre = self.embid(x_tre.unsqueeze(-1)).transpose(-2, -1)
        x_sea = self.embid(x_sea.unsqueeze(-1)).transpose(-2, -1)
        return x, x_tre, x_sea

    # trend
    @staticmethod
    def trend_model(data, t, device):
        data_len = data.shape[-1]
        T = torch.tensor([t * i for i in range(data_len)]).float()
        T = T.unsqueeze(0).unsqueeze(0).to(device)
        data = data * T
        return data

    # season
    @staticmethod
    def seasonality_model(thetas, t, device):
        p = thetas.size()[-1]
        #assert p <= thetas.shape[1], 'thetas_dim is too big.'
        p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
        s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
        s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
        s1 = s1.unsqueeze(0).unsqueeze(0).to(device)
        s2 = s2.unsqueeze(0).unsqueeze(0).to(device)
        S = torch.cat([s1, s2],dim=-1)
        thetas = thetas * S
        return thetas