import torch
import torch.nn as nn
import numpy as np


class GatedAttention(nn.Module):
    def __init__(self, hidden_dim, dim=-1, device="cuda:0"):
        super(GatedAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            # nn.Softmax(dim=dim),
            nn.Sigmoid()
        )
        self.weights = None

    def forward(self, x):
        weights = self.encoder(x)
        self.weights = weights
        return torch.mul(weights, x)

class DualMLPLayer(nn.Module):
    def __init__(self, window_size, hidden_dim, dropout=0.5):
        super(DualMLPLayer, self).__init__()
        self.block1 = nn.Sequential(
            # nn.LayerNorm(normalized_shape=window_size, elementwise_affine=False),
            nn.Linear(in_features=window_size, out_features=window_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=window_size * 2, out_features=window_size),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            # nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=False),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim),
            nn.Dropout(dropout)
        )
        self.ln1 = nn.LayerNorm(normalized_shape=window_size, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=True)
        self.ln3 = nn.LayerNorm(normalized_shape=window_size, elementwise_affine=True)
        self.ln4 = nn.LayerNorm(normalized_shape=hidden_dim, elementwise_affine=True)
        self.gat_weights_1 = None
        self.gat_weights_2 = None
        self.gat1 = GatedAttention(hidden_dim=window_size, dim=-1)
        self.gat2 = GatedAttention(hidden_dim=hidden_dim, dim=-2)

    def forward(self, x1, x2):
        # x1.shape = (b, w, f), x2.shape = (b, w, f)
        x1 = x1.transpose(-1, -2)  # x1.shape = (b, f, w)
        x1 = self.ln1(self.block1(x1) + x1)  # x1.shape = (b, f, w)
        x2 = self.ln2(self.block2(x2) + x2)  # x2.shape = (b, w, f)
        x1 = self.ln3(x1 + self.gat2(x2).transpose(-1, -2))
        x2 = self.ln4(x2 + self.gat1(x1).transpose(-1, -2))  # x2.shape = (b, f, w)
        self.gat_weights_1 = self.gat1.weights
        self.gat_weights_2 = self.gat2.weights
        return x1.transpose(-1, -2), x2


class DualMLPMixer(nn.Module):
    def __init__(self, configs,input_feature):
        super(DualMLPMixer, self).__init__()

        window_size = configs.input_length
        self.MaV = None
        self.input_embedding = nn.Linear(in_features=input_feature, out_features=configs.d_model)

        self.layers = nn.ModuleList()
        for _ in range(2):
            self.layers.append(DualMLPLayer(window_size=window_size, hidden_dim=configs.d_model, dropout=configs.dropout))
        self.out_gat1 = GatedAttention(hidden_dim=window_size)
        self.out_gat2 = GatedAttention(hidden_dim=configs.d_model, dim=-2)
        # self.fuse = nn.Linear(in_features=in_features*hidden_dim, out_features=768)
        self.output = nn.Sequential(
            # nn.Dropout(dropout),
            nn.Linear(in_features=configs.d_model*window_size, out_features=1)
        )


    def feature_extractor(self, x, idx_x=None):
        if self.MaV:
            x = self.MaV(x.transpose(-1, -2)).transpose(-1, -2)
        self.hidden_out_1 = []
        self.hidden_out_2 = []
        # x.shape = (b, w, f)
        x = self.input_embedding(x) if self.input_embedding is not None else x  # x.shape = (b, w, h)
        f1 = x
        f2 = x
        for l in self.layers:
            f1, f2 = l(f1, f2)
        f1 = self.out_gat1(f1.transpose(-1, -2))
        f2 = self.out_gat2(f2)
        f = torch.flatten(f1.transpose(-1, -2) + f2, start_dim=-2, end_dim=-1)
        return f

    def forward(self, x, label=None):
        if len(x.shape) < 4:
            x = self.feature_extractor(x)
            return None,self.output(x)


