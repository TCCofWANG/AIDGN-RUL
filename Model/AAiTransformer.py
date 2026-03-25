import torch
import torch.nn as nn
import torch.nn.functional as F
from mixture_of_experts import MoE

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class AAiTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, input_feature):
        super(AAiTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = configs.input_length
        self.pred_len = 1
        self.output_attention = False
        self.use_norm = False

        self.project = nn.Linear( 1, input_feature )
        self.project_d_model = nn.Linear(self.input_length, configs.d_model)
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.input_length, configs.d_model, configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.AAencoder = AAEncoder(
            [
                AAEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.linear = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff, bias=True),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, self.pred_len),
        )
        self.attn= nn.Linear(1,1)
        self.sensor_fc = nn.Linear(input_feature, self.pred_len, bias=True)


    def forward(self, x_enc, aging=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        aging = aging.transpose(1, 2).to( self.device ).to(torch.float64)
        aging = self.project(  aging.permute( 0, 2, 1) )
        aging = self.project_d_model(aging.permute( 0, 2, 1))
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out = self.dropout(enc_out)
        enc_out = self.AAencoder(enc_out, aging, attn_mask=None)

        # B N E -> B N S -> B S N
        fc_output = self.linear(enc_out.permute(0, 2, 1))
        rul_prediction = self.sensor_fc(fc_output.squeeze(-1))  # (B, 1)

        return None, rul_prediction  # [B, L, D]


