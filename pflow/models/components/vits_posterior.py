import torch.nn as nn
import torch

import pflow.models.components.vits_modules as modules
import pflow.models.components.commons as commons

class PosteriorEncoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)),
                                 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        # m, logs = torch.split(stats, self.out_channels, dim=1)
        # z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        # z = m * x_mask
        return stats, x_mask
