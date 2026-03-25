import torch
import torch.nn as nn
import torch.nn.functional as F
from mixture_of_experts import MoE

from layers.Transformer_EncDec import Encoder, EncoderLayer, AAEncoderLayer, AAEncoder
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
        self.scale = None
        self.input_feature = input_feature
        self.inf = -2**32 + 1

        self.layer_norm = LayerNorm(configs.d_model)
        self.enc_embedding = DataEmbedding_inverted(configs.input_length, configs.d_model, configs.dropout)
        # self.project = nn.Linear( 1, input_feature )
        self.project_d_model = nn.Linear(self.input_length, configs.d_model)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


        self.dropout = nn.Dropout(configs.dropout)

        self.decoder = nn.Sequential( nn.Linear( configs.d_model, input_feature*4, bias=True ),
                                      nn.GELU(),
                                      # nn.Linear( args.fc_layer_dim, args.fc_layer_dim ), ACTIVATION_MAP[self.fc_activation]()( inplace = True ),
                                      nn.Linear( input_feature*4, 1 ) )

        self.channel_decoder = nn.Sequential( nn.Linear( input_feature, input_feature*4, bias=True ),
                                              nn.GELU(),
                                              nn.Linear( input_feature*4, 1 ) )
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    def encode_age(self, age_CI):
        aging = age_CI.transpose(1, 2).to( self.device ).to(torch.float64)
        aging = aging.repeat( 1, self.input_feature, 1 ) #self.project(  aging.permute( 0, 2, 1) )
        aging = self.project_d_model(aging)
        aging = self.layer_norm(aging)
        return aging

    def encode_sensor(self, sensorX):
        sensor_features = self.enc_embedding(sensorX, None) # covariates (e.g timestamp) can be also embedded as tokens
        sensor_features = self.dropout(sensor_features)
        sensor_features = self.layer_norm(sensor_features)
        return sensor_features

    def forward(self, x_enc, aging=None):

        sensor_features = self.encode_sensor(x_enc)
        aging_features = self.encode_age(aging)

        # normalized features
        sensor_features = sensor_features / sensor_features.norm(dim=1, keepdim=True)
        aging_features = aging_features / aging_features.norm(dim=1, keepdim=True)

        mix1 = sensor_features @ aging_features.transpose(-1, -2) @ sensor_features
        mix2 =  aging_features @ sensor_features.transpose(-1, -2) @ aging_features
        logits_per_window =  mix1 + mix2

        # B N E -> B N S -> B S N
        fc_output = self.decoder(logits_per_window)
        rul_prediction = self.channel_decoder(fc_output.squeeze(-1))  # self.regressor(fc_output)

        return None, rul_prediction  # [B, L, D]

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, a):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, a, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, a):
        x = x + self.attention(self.ln_1(x), self.ln_1(a))
        x = x + self.mlp(self.ln_2(x))
        return x
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float64))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
