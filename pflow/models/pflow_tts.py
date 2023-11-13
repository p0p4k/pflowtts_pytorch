import datetime as dt
import math
import random

import torch
import torch.nn.functional as F

from pflow import utils
from pflow.models.baselightningmodule import BaseLightningClass
from pflow.models.components.flow_matching import CFM
from pflow.models.components.speech_prompt_encoder import TextEncoder
from pflow.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)
from pflow.models.components import commons
from pflow.models.components.aligner import Aligner, ForwardSumLoss, BinLoss

log = utils.get_pylogger(__name__)

class pflowTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        prompt_size=264,
        optimizer=None,
        scheduler=None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        
        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.prompt_size = prompt_size
        speech_in_channels = n_feats

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            speech_in_channels,
        )

        # self.aligner = Aligner(
        #     dim_in=encoder.encoder_params.n_feats,
        #     dim_hidden=encoder.encoder_params.n_feats,
        #     attn_channels=encoder.encoder_params.n_feats,
        #     )
        
        # self.aligner_loss = ForwardSumLoss()
        # self.bin_loss = BinLoss()
        # self.aligner_bin_loss_weight = 0.0

        self.decoder = CFM(
            in_channels=encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
        )

        self.proj_prompt = torch.nn.Conv1d(encoder.encoder_params.n_channels, self.n_feats, 1)

        self.update_data_statistics(data_statistics)

    @torch.inference_mode()
    def synthesise(self, x, x_lengths, prompt, n_timesteps, temperature=1.0, length_scale=1.0):

        # For RTF computation
        t = dt.datetime.now()
        assert prompt is not None, "Prompt must be provided for synthesis"
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, prompt)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    def forward(self, x, x_lengths, y, y_lengths, prompt=None, cond=None, **kwargs):        
        if prompt is None:
            prompt_slice, ids_slice = commons.rand_slice_segments(
                        y, y_lengths, self.prompt_size
                    )
        else:
            prompt_slice = prompt
        mu_x, logw, x_mask = self.encoder(x, x_lengths, prompt_slice)
       
        y_max_length = y.shape[-1]
        
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        
        with torch.no_grad():
        # negative cross-entropy
            s_p_sq_r = torch.ones_like(mu_x) # [b, d, t]
            # s_p_sq_r = torch.exp(-2 * logx) 
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi)- torch.zeros_like(mu_x), [1], keepdim=True
            )
            # neg_cent1 = torch.sum(
            #     -0.5 * math.log(2 * math.pi) - logx, [1], keepdim=True
            #     ) # [b, 1, t_s]
            neg_cent2 = torch.einsum("bdt, bds -> bts", -0.5 * (y**2), s_p_sq_r)
            neg_cent3 = torch.einsum("bdt, bds -> bts", y, (mu_x * s_p_sq_r))
            neg_cent4 = torch.sum(
                -0.5 * (mu_x**2) * s_p_sq_r, [1], keepdim=True
            )  
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            from pflow.utils.monotonic_align import maximum_path
            attn = (
                maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
            )

        logw_ = torch.log(1e-8 + attn.sum(2)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # aln_hard, aln_soft, aln_log, aln_mask = self.aligner(
        #     mu_x.transpose(1,2), x_mask, y, y_mask
        #     )
        # attn = aln_mask.transpose(1,2).unsqueeze(1)
        # align_loss = self.aligner_loss(aln_log, x_lengths, y_lengths)
        # if self.aligner_bin_loss_weight > 0.:
        #     align_bin_loss = self.bin_loss(aln_mask, aln_log, x_lengths) * self.aligner_bin_loss_weight
        #     align_loss = align_loss + align_bin_loss
        # dur_loss = F.l1_loss(logw, attn.sum(2))
        # dur_loss = dur_loss + align_loss
        
        # Align encoded text with mel-spectrogram and get mu_y segment
        attn = attn.squeeze(1).transpose(1,2)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        y_loss_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        if prompt is None:
            for i in range(y.size(0)):  
                y_loss_mask[i,:,ids_slice[i]:ids_slice[i] + self.prompt_size] = 0 
        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(x1=y.detach(), mask=y_mask, mu=mu_y, cond=cond, loss_mask=y_loss_mask)
        
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_loss_mask)
        prior_loss = prior_loss / (torch.sum(y_loss_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss, attn