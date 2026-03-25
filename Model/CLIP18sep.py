import torch
import torch.nn as nn
import torch.nn.functional as F
from mixture_of_experts import MoE

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class CLIP(nn.Module):
    def __init__(self, configs, input_feature):
        super(CLIP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = configs.input_length
        self.pred_len = 1
        self.output_attention = False
        self.use_norm = False
        self.scale = None
        self.input_feature = input_feature
        self.inf = -2**32 + 1

        self.enc_embedding = DataEmbedding_inverted(configs.input_length, configs.d_model, configs.dropout)
        self.project = nn.Linear( 1, input_feature )
        self.project_d_model = nn.Linear(self.input_length, configs.d_model)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        self.dropout = nn.Dropout(configs.dropout)

        self.linear = nn.Sequential(
            nn.Linear(input_feature, configs.d_ff, bias=True),
            nn.Softmax(dim=1),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, self.pred_len),
        )
        self.sensor_fc = nn.Linear(input_feature, self.pred_len, bias=True)

    def encode_age(self, age):
        aging = age.transpose(1, 2).to( self.device ).to(torch.float64)
        aging = self.project(  aging.permute( 0, 2, 1) )
        aging = self.project_d_model(aging.permute( 0, 2, 1))
        return aging

    def forward(self, x_enc, aging=None):
        scale = self.scale or 1. / np.sqrt ( self.input_feature )
        sensor_features = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        sensor_features = self.dropout(sensor_features)

        aging_features = self.encode_age(aging)
        #aging_features = self.dropout(aging_features)

        # normalized features
        sensor_features = sensor_features / sensor_features.norm(dim=1, keepdim=True)
        aging_features = aging_features / aging_features.norm(dim=1, keepdim=True)


        mix1 = sensor_features @ aging_features.transpose(-1, -2)
        mix2 =  aging_features @ sensor_features.transpose(-1, -2)
        enc_out =  mix1 + mix2

        # B N E -> B N S -> B S N
        fc_output = self.linear(enc_out.permute(0, 2, 1))
        rul_prediction = self.sensor_fc(fc_output.squeeze(-1))  # (B, 1)

        return None, rul_prediction  # [B, L, D]


