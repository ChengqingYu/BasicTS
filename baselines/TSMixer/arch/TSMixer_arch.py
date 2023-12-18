import torch
import torch.nn as nn

from .block import RevIN
from .block import ResBlock


class TSMixer(nn.Module):
    """Implementation of TSMixerRevIN."""

    def __init__(self, input_len, vare_num, pred_len, n_block, dropout, ff_dim):
        super(TSMixer, self).__init__()

        self.rev_norm = RevIN(vare_num)

        self.res_blocks = nn.ModuleList([ResBlock(input_len,vare_num, dropout, ff_dim) for _ in range(n_block)])

        self.linear = nn.Linear(input_len, pred_len)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool,
                **kwargs) -> torch.Tensor:

        x = history_data[:,:,:,0]

        x = self.rev_norm(x, 'norm')

        for res_block in self.res_blocks:
            x = res_block(x)

        x = torch.transpose(x, 1, 2)
        x = self.linear(x)
        x = torch.transpose(x, 1, 2)

        x = self.rev_norm(x, 'denorm')

        return x.unsqueeze(-1)
