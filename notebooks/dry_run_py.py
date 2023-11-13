import sys
sys.path.append('..')

from pflow.models.pflow_tts import pflowTTS

import torch
from dataclasses import dataclass

@dataclass
class DurationPredictorParams:
    filter_channels_dp: int
    kernel_size: int
    p_dropout: float

@dataclass
class EncoderParams:
    n_feats: int
    n_channels: int
    filter_channels: int
    filter_channels_dp: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    spk_emb_dim: int
    n_spks: int
    prenet: bool

@dataclass
class CFMParams:
    name: str
    solver: str
    sigma_min: float

# Example usage
duration_predictor_params = DurationPredictorParams(
    filter_channels_dp=256,
    kernel_size=3,
    p_dropout=0.1
)

encoder_params = EncoderParams(
    n_feats=80,
    n_channels=192,
    filter_channels=768,
    filter_channels_dp=256,
    n_heads=2,
    n_layers=6,
    kernel_size=3,
    p_dropout=0.1,
    spk_emb_dim=64,
    n_spks=1,
    prenet=True
)

cfm_params = CFMParams(
    name='CFM',
    solver='euler',
    sigma_min=1e-4
)

@dataclass
class EncoderOverallParams:
    encoder_type: str
    encoder_params: EncoderParams
    duration_predictor_params: DurationPredictorParams

encoder_overall_params = EncoderOverallParams(
    encoder_type='RoPE Encoder',
    encoder_params=encoder_params,
    duration_predictor_params=duration_predictor_params
)

model = pflowTTS(
    n_vocab=100,
    n_feats=80,
    encoder=encoder_overall_params,
    decoder=None,
    cfm=cfm_params,
    data_statistics=None,
)

x = torch.randint(0, 100, (4, 20))
x_lengths = torch.randint(10, 20, (4,))
y = torch.randn(4, 80, 500)
y_lengths = torch.randint(300, 500, (4,))

dur_loss, prior_loss, diff_loss, attn = model(x, x_lengths, y, y_lengths)

print(dur_loss, prior_loss, diff_loss)