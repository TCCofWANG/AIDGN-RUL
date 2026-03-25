from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from layers.Embed import DataEmbedding_inverted


class CLIP(nn.Module):
    def __init__(self, configs, input_feature):
        super(CLIP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = configs.input_length
        self.pred_len = 1
        self.input_feature = input_feature

        self.enc_embedding = DataEmbedding_inverted(
            configs.input_length, configs.d_model, configs.dropout
        )

        # projection layers for aging
        self.project = nn.Linear(1, input_feature)
        self.project_d_model = nn.Linear(self.input_length, configs.d_model)

        # scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # normalization + dropout
        self.dropout = nn.Dropout(configs.dropout)
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.layer_norm2 = nn.LayerNorm(input_feature)

        # fusion MLP head LeakyReLU FD003 9.6 158
        self.fc = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_ff, bias=True),
            nn.LeakyReLU(),
            nn.Linear(configs.d_ff, configs.d_ff, bias=True),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, self.pred_len),
        )

        # final regression head
        self.sensor_fc = nn.Sequential(
            nn.Linear(input_feature, input_feature * 4, bias=True),
            nn.LeakyReLU(),
            nn.Linear(input_feature * 4, self.pred_len),
        )

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode_age(self, age):
        aging = age.transpose(1, 2).to(self.device).to(torch.float32)  # use float32 for consistency
        aging = self.project(aging.permute(0, 2, 1))
        aging = self.project_d_model(aging.permute(0, 2, 1))
        return aging

    def forward(self, x_enc, aging=None):
        sensor_features = self.enc_embedding(x_enc, None)
        sensor_features = self.layer_norm(self.dropout(sensor_features))

        aging_features = self.encode_age(aging)
        aging_features = self.layer_norm(self.dropout(aging_features))

        # normalize features
        sensor_features = sensor_features / (sensor_features.norm(dim=-1, keepdim=True) + 1e-6)
        aging_features = aging_features / (aging_features.norm(dim=-1, keepdim=True) + 1e-6)


        # bi-directional similarity fusion + residual
        d_k = sensor_features.size(-1)
        mix1 = (sensor_features @ aging_features.transpose(-1, -2)) / np.sqrt(d_k)
        mix2 = (aging_features @ sensor_features.transpose(-1, -2)) / np.sqrt(d_k)

        # identity = torch.eye(sensor_features.size(1), device=sensor_features.device).unsqueeze(0)
        enc_out = (mix1 + mix2) @ sensor_features

        enc_out = self.layer_norm(enc_out)

        # prediction head
        fc_output = self.fc(enc_out)

        rul_prediction = self.sensor_fc(fc_output.permute(0, 2, 1))
        rul_prediction = torch.abs(rul_prediction.squeeze(-1))
        return None, rul_prediction



class AttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super(AttentionPooling, self).__init__()
        # learnable scoring function
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: [B, T, d_model]
        scores = self.attn(x)  # [B, T, 1]
        weights = F.softmax(scores, dim=1)  # normalize across time
        pooled = (x * weights).sum(dim=1)   # weighted sum → [B, d_model]
        return pooled