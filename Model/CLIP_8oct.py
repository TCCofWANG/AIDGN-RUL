import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CLIP(nn.Module):
    def __init__(self, configs, input_feature):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_length = configs.input_length
        self.pred_len = 1
        self.input_feature = input_feature
        d_model = configs.d_model

        # Embedding
        from layers.Embed import DataEmbedding_inverted
        self.enc_embedding = DataEmbedding_inverted(
            configs.input_length, d_model, configs.dropout
        )

        # Age projection
        self.project = nn.Linear(1, input_feature)
        self.project_d_model = nn.Linear(self.input_length, d_model)

        # Attention projections for cross fusion
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # LayerNorm + dropout
        self.dropout = nn.Dropout(configs.dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(input_feature)

        # Temporal attention pooling
        self.temporal_pool = AttentionPooling(d_model)

        # Prediction head
        self.fc = nn.Sequential(
            nn.Linear(d_model, configs.d_ff),
            nn.LeakyReLU(),
            nn.Linear(configs.d_ff, configs.d_ff),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, self.pred_len),
        )

        # Regression head
        self.sensor_fc = nn.Sequential(
            nn.Linear(input_feature, input_feature * 4),
            nn.LeakyReLU(),
            nn.Linear(input_feature * 4, self.pred_len),
        )

        # Init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode_age(self, age):
        # age: [B, T, 1]
        aging = age.transpose(1, 2).to(self.device).to(torch.float32)
        aging = self.project(aging.permute(0, 2, 1))
        aging = self.project_d_model(aging.permute(0, 2, 1))
        return aging

    def cross_attention(self, sensor, age):
        # sensor, age: [B, T, d]
        Q = self.query_proj(sensor)
        K = self.key_proj(age)
        V = self.value_proj(age)

        attn = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(Q.size(-1))
        attn = F.softmax(attn, dim=-1)
        fused = torch.bmm(attn, V)
        return fused + sensor  # residual connection

    def forward(self, x_enc, aging=None):
        # Encode sensor
        sensor_features = self.enc_embedding(x_enc, None)
        sensor_features = self.norm1(self.dropout(sensor_features))

        # Encode aging
        aging_features = self.encode_age(aging)
        aging_features = self.norm2(self.dropout(aging_features))

        # Cross-attention fusion (sensor queries age)
        fused = self.cross_attention(sensor_features, aging_features)
        fused = self.norm1(fused)

        # Temporal pooling to summarize degradation
        pooled = self.temporal_pool(fused)  # [B, d_model]

        # Prediction head
        fc_output = self.fc(pooled)    # [B, C, 1]
        rul_prediction = self.sensor_fc(fc_output.permute(0, 2, 1))
        rul_prediction = torch.abs(rul_prediction.squeeze(-1))
        return None, rul_prediction


class AttentionPooling(nn.Module):
    def __init__(self, d_model, hidden_dim=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        scores = self.attn(x)              # [B, T, 1]
        weights = F.softmax(scores, dim=1)
        pooled = (x * weights) / np.sqrt(x.size(-1))  # [B, d_model]
        return pooled
