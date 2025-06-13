# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch
import torch.nn as nn

from super_monotonic_align import maximum_path
from models.tts.matchatts.base import BaseModule
# from models.tts.matchatts.text_encoder import TextEncoder
from models.tts.matchatts.flow_matching import CFM
from models.tts.matchatts.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
)
from models.tts.modules.style_encoder import StyleEncoder
from models.tts.matchatts.text_encoder import ConvReluNorm, DurationPredictor
from models.commons.layers import Embedding
from utils.audio.pitch.utils import denorm_f0, f0_to_coarse, norm_f0
from utils.audio.cwt import cwt2f0, get_lf0_cwt
import numpy as np
from speech_synthesis.models.commons.nar_tts_modules import PitchPredictor
from models.tts.matchatts.text_encoder import MultiHeadAttention, LayerNorm, FFN, CrossAttention


class Encoder(BaseModule):
    def __init__(
        self,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size=1,
        p_dropout=0.0,
        window_size=None,
        f_cond_dim=None,
        **kwargs
    ):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.drop = torch.nn.Dropout(p_dropout)
        self.attn_layers = torch.nn.ModuleList()
        self.norm_layers_1 = torch.nn.ModuleList()
        self.cross_attn_layers = torch.nn.ModuleList()
        self.norm_layers_2 = torch.nn.ModuleList()
        self.ffn_layers = torch.nn.ModuleList()
        self.norm_layers_3 = torch.nn.ModuleList()
        for _ in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    window_size=window_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.cross_attn_layers.append(
                CrossAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_3.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, wf, wf_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        cross_attn_mask = x_mask.unsqueeze(2) * wf_mask.unsqueeze(-1)

        for i in range(self.n_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.cross_attn_layers[i](x, wf, cross_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_3[i](x + y)
        x = x * x_mask
        return x

class TextEncoder(BaseModule):
    def __init__(
        self,
        n_vocab,
        n_feats,
        n_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size=None,
        spk_emb_dim=64,
        n_spks=1,
        f_cond_dim=64,
    ):
        super(TextEncoder, self).__init__()
        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.n_channels = n_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks

        self.emb = Embedding(n_vocab, n_channels, 0)

        self.prenet = ConvReluNorm(
            n_channels, n_channels, n_channels, kernel_size=5, n_layers=3, p_dropout=0.5
        )

        self.expand_layer = torch.nn.Conv1d(
            f_cond_dim,
            n_channels + spk_emb_dim + f_cond_dim,
            1,
        )

        self.encoder = Encoder(
            n_channels
            + spk_emb_dim
            + f_cond_dim,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            f_cond_dim=f_cond_dim
        )

        self.proj_m = torch.nn.Conv1d(
            n_channels
            + spk_emb_dim
            + f_cond_dim,
            n_feats,
            1,
        )
        self.proj_w = DurationPredictor(
            n_channels
            + spk_emb_dim
            + f_cond_dim * 2,
            filter_channels_dp,
            kernel_size,
            p_dropout,
        )


    def forward(self, x, x_lengths, f_conds, dur_conds, wf, wf_lengths, spk=None):
        x = self.emb(x) * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1).contiguous()
        wf = torch.transpose(wf, 1, -1).contiguous()
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        wf_mask = torch.unsqueeze(sequence_mask(wf_lengths, wf.size(2)), 1).to(f_conds.dtype)

        x = self.prenet(x, x_mask)

        x = torch.cat([x, spk.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = torch.cat([x, f_conds.transpose(1, -1)], dim=1)

        wf = self.expand_layer(wf) * wf_mask

        x = self.encoder(x, x_mask, wf, wf_mask)

        mu = self.proj_m(x) * x_mask

        x_dp = torch.detach(x)
        x_dp = torch.cat([x_dp, dur_conds.transpose(1, -1)], dim=1)
        logw = self.proj_w(x_dp, x_mask)

        return mu, logw, x_mask


class FillerSpeech(BaseModule):
    def __init__(self, dict_size, hparams, out_dims=80):
        super(FillerSpeech, self).__init__()
        self.n_vocab = dict_size
        self.n_spks = hparams["n_spks"]
        self.spk_emb_dim = hparams["spk_emb_dim"]
        self.n_enc_channels = hparams["n_enc_channels"]
        self.filter_channels = hparams["filter_channels"]
        self.filter_channels_dp = hparams["filter_channels_dp"]
        self.n_heads = hparams["n_heads"]
        self.n_enc_layers = hparams["n_enc_layers"]
        self.enc_kernel = hparams["enc_kernel"]
        self.enc_dropout = hparams["enc_dropout"]
        self.window_size = hparams["window_size"]
        self.n_feats = out_dims
        self.hidden_size = hparams["hidden_size"]
        self.predictor_grad = hparams["predictor_grad"]
        self.cwt_std_scale = hparams["cwt_std_scale"]


        self.cfm_params = {
            "solver": hparams["solver"],
            "sigma_min": hparams["sigma_min"],
        }
        self.decoder_params = {
            "channels": hparams["channels"],
            "dropout": hparams["dropout"],
            "attention_head_dim": hparams["attention_head_dim"],
            "n_blocks": hparams["n_blocks"],
            "num_mid_blocks": hparams["num_mid_blocks"],
            "num_heads": hparams["num_heads"],
            "act_fn": hparams["act_fn"],
            "down_block_type": hparams["down_block_type"],
            "mid_block_type": hparams["mid_block_type"],
            "up_block_type": hparams["up_block_type"],
        }

        self.spk_emb = StyleEncoder(self.n_feats, 256, self.spk_emb_dim)
        self.encoder = TextEncoder(
            self.n_vocab,
            self.n_feats,
            self.n_enc_channels,
            self.filter_channels,
            self.filter_channels_dp,
            self.n_heads,
            self.n_enc_layers,
            self.enc_kernel,
            self.enc_dropout,
            self.window_size,
            spk_emb_dim=self.spk_emb_dim,
            f_cond_dim=hparams["f_cond_dim"]
        )
        self.decoder = CFM(
            in_channels=2 * self.n_feats,
            out_channel=self.n_feats,
            cfm_params=self.cfm_params,
            decoder_params=self.decoder_params,
            n_spks=self.n_spks,
            spk_emb_dim=self.spk_emb_dim,
        )

        self.f_cond_embedding = Embedding(
            num_embeddings=hparams["f_cond_num"], embedding_dim=hparams["f_cond_dim"], padding_idx=0)
        self.wf_embedding = Embedding(
            num_embeddings=hparams["f_cond_num"], embedding_dim=hparams["f_cond_dim"], padding_idx=0)
        self.dur_cond_embedding = Embedding(
            num_embeddings=hparams["dur_cond_num"], embedding_dim=hparams["f_cond_dim"], padding_idx=0)


        self.pitch_embed = Embedding(300, self.n_feats, 0)
        self.pitch_predictor = PitchPredictor(
            self.n_feats,
            n_chans=self.hidden_size,
            n_layers=hparams["predictor_layers"],
            dropout_rate=hparams["predictor_dropout"],
            odim=11,
            kernel_size=hparams["predictor_kernel"],
        )
        self.cwt_stats_layers = nn.Sequential(
            nn.Linear(self.n_feats, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )


    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        spk=None,
        spk_lengths=None,
        f_conds=None,
        dur_conds=None,
        wf=None,
        wf_lengths=None,
        length_scale=1.0,
    ):

        x, x_lengths, spk, spk_lengths, f_conds, dur_conds, wf, wf_lengths = self.relocate_input(
            [x, x_lengths, spk, spk_lengths, f_conds, dur_conds, wf, wf_lengths])

        spk = self.spk_emb(spk, spk_lengths)


        f_conds = self.f_cond_embedding(f_conds)
        dur_conds = self.dur_cond_embedding(dur_conds)

        wf = self.wf_embedding(wf)
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths,
                                          f_conds=f_conds, dur_conds=dur_conds,
                                          wf=wf, wf_lengths=wf_lengths, spk=spk)


        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        mel2ph = torch.ones(mu_y.shape[0], y_max_length_, dtype=mu_y.dtype, device=mu_y.device)
        mu_y = mu_y + self.forward_pitch(mu_y, mel2ph=mel2ph)

        encoder_outputs = mu_y[:, :, :y_max_length]

        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, temperature, spk)

        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        ret = {}
        ret["encoder_outputs"] = encoder_outputs.transpose(1, 2)
        ret["mel_out"] = decoder_outputs.transpose(1, 2)
        ret["attn"] = attn[:, :, :y_max_length].squeeze(1)
        return ret

    def compute_loss(self, x, x_lengths, y, y_lengths, f_conds, dur_conds, wf, wf_lengths,
                     f0, uv, mel2ph, spk=None, spk_lengths=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, spk, spk_lengths, f_conds, dur_conds, wf, wf_lengths = self.relocate_input(
            [x, x_lengths, spk, spk_lengths, f_conds, dur_conds, wf, wf_lengths])

        spk = self.spk_emb(spk, spk_lengths)

        f_conds = self.f_cond_embedding(f_conds)
        dur_conds = self.dur_cond_embedding(dur_conds)
        wf = self.wf_embedding(wf)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, f_conds=f_conds, dur_conds=dur_conds,
                                          wf=wf, wf_lengths=wf_lengths, spk=spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask

        dur_loss = duration_loss(logw, logw_, x_lengths)


        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        pitch_temp = {}
        mu_y = mu_y + self.forward_pitch(mu_y, f0, uv, mel2ph, pitch_temp)
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)

            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            mu_y_cut = torch.zeros(
                mu_y.shape[0], self.n_feats, out_size, dtype=mu_y.dtype, device=mu_y.device
            )

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                mu_y_cut[i, :, :y_cut_length] = mu_y[i, :, cut_lower:cut_upper]


            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)

            y = y_cut
            y_mask = y_cut_mask
            mu_y = mu_y_cut


        diff_loss, _ = self.decoder.compute_loss(
            x1=y, mask=y_mask, mu=mu_y, spks=spk  # , cond=cond
        )


        ret = {}
        ret["dur_loss"] = dur_loss
        ret["prior_loss"] = prior_loss
        ret["diff_loss"] = diff_loss

        return ret, pitch_temp

    def forward_pitch(self, decoder_inp, f0=None, uv=None, mel2ph=None, ret={}):
        # Input: [B, T, H] ? H가 D
        decoder_inp = decoder_inp.transpose(1, 2)
        decoder_inp = decoder_inp.detach() + self.predictor_grad * (decoder_inp - decoder_inp.detach())
        pitch_padding = mel2ph == 0
        ret['cwt'] = cwt_out = self.pitch_predictor(decoder_inp)
        stats_out = self.cwt_stats_layers(decoder_inp.mean(1))  # [B, 2]
        mean = ret['f0_mean'] = stats_out[:, 0]
        std = ret['f0_std'] = stats_out[:, 1]
        cwt_spec = cwt_out[:, :, :10]
        if f0 is None:
            std = std * self.cwt_std_scale
            f0 = self.cwt2f0_norm(cwt_spec, mean, std, mel2ph)
            assert cwt_out.shape[-1] == 11
            uv = cwt_out[:, :, -1] > 0
        ret['f0_denorm'] = f0_denorm = denorm_f0(f0, uv,
                                                 pitch_padding=pitch_padding)
        pitch = f0_to_coarse(f0_denorm)  # start from 0
        pitch_embed = self.pitch_embed(pitch)
        return pitch_embed.transpose(1, 2)


    def cwt2f0_norm(self, cwt_spec, mean, std, mel2ph):
        _, cwt_scales = get_lf0_cwt(np.ones(10))
        f0 = cwt2f0(cwt_spec, mean, std, cwt_scales)
        f0 = torch.cat([f0] + [f0[:, -1:]] * (mel2ph.shape[1] - f0.shape[1]), 1)
        f0_norm = norm_f0(f0, None)
        return f0_norm
