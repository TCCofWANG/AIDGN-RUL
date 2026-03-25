import torch
import torch.nn as nn
import torch.nn.functional as F
from mixture_of_experts import MoE

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class CLIP(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, input_feature):
        super(CLIP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = configs.input_length
        self.pred_len = 1
        self.output_attention = False
        self.use_norm = False

        self.enc_embedding = DataEmbedding_inverted(configs.input_length, configs.d_model, configs.dropout)
        self.project = nn.Linear( 1, input_feature )
        self.project_d_model = nn.Linear(self.input_length, configs.d_model)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        self.dropout = nn.Dropout(configs.dropout)
        self.linear = nn.Sequential(
            nn.Linear(input_feature, configs.d_model*2, bias=True),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model*2, self.pred_len),
        )

    def encode_sensor(self, x_enc):
        return self
    def encode_age(self, age):
        aging = age.transpose(1, 2).to( self.device ).to(torch.float64)
        aging = self.project(  aging.permute( 0, 2, 1) )
        aging = self.project_d_model(aging.permute( 0, 2, 1))
        return aging

    def forward(self, x_enc, aging=None):
        sensor_features = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        sensor_features = self.dropout(sensor_features)

        aging_features = self.encode_age(aging)

        # normalized features
        sensor_features = sensor_features / sensor_features.norm(dim=1, keepdim=True)
        aging_features = aging_features / aging_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_window = logit_scale * sensor_features @ aging_features.t()
        # logits_per_text = logits_per_window.t()
        mix1 = sensor_features @ aging_features.transpose(-1, -2)
        mix2 =  aging_features@ sensor_features.transpose(-1, -2)
        logits_per_window =  mix1+mix2 #F.log_softmax(self.linear(aging_features))
        # out,_ = self.moe(enc_out.permute( 0, 2, 1 ))
        # out = self.linear(out)  # shape: (batch_size, 1)
        # rul_prediction = torch.abs(out[:, -1, :])

        # B N E -> B N S -> B S N
        fc_output = self.linear(logits_per_window.permute(0, 2, 1))
        fc_output = self.dropout(fc_output)
        rul_prediction = torch.abs(fc_output[:, -1, :])  # self.regressor(fc_output)

        return None, rul_prediction  # [B, L, D]


