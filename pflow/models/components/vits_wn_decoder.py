import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import pflow.models.components.vits_modules as modules
import pflow.models.components.commons as commons

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class VitsWNDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0,
                 pe_scale=1000
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.pe_scale = pe_scale    
        self.time_pos_emb = SinusoidalPosEmb(hidden_channels * 2)
        dim = hidden_channels * 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels * 2,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels * 2, out_channels, 1)

    def forward(self, x, x_mask, mu, t, *args, **kwargs):
        # x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)),
        #                          1).to(x.dtype)
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        x = self.pre(x) * x_mask
        mu = self.pre(mu)
        x = torch.cat((x, mu), dim=1)
        x = self.enc(x, x_mask, g=t)
        stats = self.proj(x) * x_mask

        return stats
