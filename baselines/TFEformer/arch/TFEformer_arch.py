import torch
from torch import nn

from .Decompo_block import decomp_tre_sea
from .temporal_attention_block import Time_att
from .Fourie_attention_block import four_att
from .Frequency_attention_block import freq_att
from .revin import RevIN


class TFEformer(nn.Module):
    def __init__(self, Input_len, out_len, num_id,dropout, num_hid,muti_head,T, IF_REVIN):
        """
        Input_len: Historical observation length
        out_len：Future value length
        num_id：number of time series
        num_hid: embiding size
        muti_head：number of muti-head attention
        T:Hyperparameters of trend and seasonal decomposition
        """
        super(TFEformer, self).__init__()

        # data solve
        self.IF_REVIN = IF_REVIN
        self.RevIN = RevIN(num_id)
        self.decomp_tre_sea = decomp_tre_sea(Input_len, num_hid, T)

        # encode
        self.Time_att = Time_att(Input_len, muti_head,dropout)
        self.four_att = four_att(Input_len, muti_head,dropout)
        self.freq_att = freq_att(Input_len, muti_head,dropout)
        self.laynorm = nn.LayerNorm([num_id,num_hid,Input_len])

        # decode
        self.decod = nn.Linear(num_hid,1)
        self.output = nn.Linear(Input_len,out_len)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        # Input [B,H,N,1]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N,1]: B is batch size. N is the number of variables. L is the future length

        if self.IF_REVIN:
            x = history_data[:, :, :, 0]
            x = self.RevIN(x,'norm').transpose(-2,-1)
        else:
            x = history_data[:, :, :, 0].transpose(-2,-1)

        ### encode
        x, x_tre, x_sea = self.decomp_tre_sea(x, x.device)
        x = self.Time_att(x) + self.four_att(x_tre) + self.freq_att(x_sea)
        x = self.laynorm(x)

        ### output
        x = x.transpose(-2, -1)
        x = self.decod(x)
        x = x.squeeze(-1)
        x = self.output(x).transpose(-2,-1)

        if self.IF_REVIN:
            x = self.RevIN(x, 'denorm')
            x = x.unsqueeze(-1)
        else:
            x = x.unsqueeze(-1)
        return x
