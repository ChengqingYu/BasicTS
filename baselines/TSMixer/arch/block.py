import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block of TSMixer."""

    def __init__(self, input_len,vare_num, dropout, ff_dim):
        super(ResBlock, self).__init__()

        # Temporal Linear
        self.norm1 = nn.BatchNorm1d(input_len * vare_num)
        self.linear1 = nn.Linear(input_len, input_len)
        self.dropout1 = nn.Dropout(dropout)

        # Feature Linear
        self.norm2 = nn.BatchNorm1d(input_len * vare_num)
        self.linear2 = nn.Linear(vare_num, ff_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.linear3 = nn.Linear(ff_dim, vare_num)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        inputs = x.clone()

        # Temporal Linear
        x = self.norm1(torch.flatten(x, 1, -1)).reshape(x.shape)
        x = torch.transpose(x, 1, 2)
        x = F.gelu(self.linear1(x))
        x = torch.transpose(x, 1, 2)
        x = self.dropout1(x)

        res = x + inputs

        # Feature Linear
        x = self.norm2(torch.flatten(res, 1, -1)).reshape(res.shape)
        x = F.gelu(self.linear2(x))
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.dropout3(x)

        return x + res


# https://github.com/ts-kim/RevIN/blob/master/RevIN.py
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x