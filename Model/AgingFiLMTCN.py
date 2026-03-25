import torch
import torch.nn as nn
import torch.nn.functional as F
from mixture_of_experts import MoE

from layers.Transformer_EncDec import Encoder, EncoderLayer, AAEncoderLayer, AAEncoder
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class AgingFiLMTCN(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, input_feature):
        super(AgingFiLMTCN, self).__init__()
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
            nn.Linear(configs.d_model, configs.d_model, bias=True),
            nn.ReLU(),
            # nn.Linear(configs.d_model, configs.d_ff, bias=True),
            nn.Softmax(dim=1),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_ff, self.pred_len),
        )
        self.attn= nn.Linear(1,1)
        # self.decoder = nn.Sequential( nn.Linear( input_feature, configs.d_ff ),
        #                               nn.GELU(),
        #                               # nn.Linear( args.fc_layer_dim, args.fc_layer_dim ), ACTIVATION_MAP[self.fc_activation]()( inplace = True ),
        #                               nn.Linear( configs.d_ff, 1 ) )
        #
        # self.channel_decoder = nn.Sequential( nn.Linear( input_feature, configs.d_ff ),
        #                                       nn.GELU(),
        #                                       nn.Linear( configs.d_ff, 1 ) )

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



        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        # logits_per_window = logit_scale * sensor_features @ aging_features.t()
        # logits_per_text = logits_per_window.t()
        mix1 = sensor_features @ aging_features.transpose(-1, -2)
        mix2 =  aging_features @ sensor_features.transpose(-1, -2)
        logits_per_window =  mix1 + mix2

        # dist_scores = torch.cdist(sensor_features, aging_features, p=2)
        # dist_scores = self.dropout(dist_scores)
        # # Compute inverse Gaussian kernel
        # inverse_scores = 1 - torch.exp(-dist_scores)  # Higher scores for larger distances
        # scores = self.dropout(inverse_scores)
        # A = torch.softmax(scale * scores, dim=-1)
        # A = self.dropout(A)
        #
        # V = torch.einsum('bls,bsd->bld', A, sensor_features)

        # fc_output = self.dropout( logits_per_window )
        # decoder_output = self.decoder( fc_output )
        # decoder_output_p = decoder_output.permute( 0, 2, 1 )
        # cd_output = self.channel_decoder( decoder_output_p ).permute( 0, 2, 1 )
        # rul_prediction = torch.abs(cd_output.squeeze(-1))

        # B N E -> B N S -> B S N
        fc_output = self.linear(logits_per_window.permute(0, 2, 1))
        fc_output = self.dropout(fc_output)
        scores = self.attn(fc_output)                  # (B, num_sensors, 1)
        weights = torch.softmax(scores, dim=1)         # (B, num_sensors, 1)
        rul_prediction = torch.sum(weights * fc_output, dim=1)  # (B, 1)

        return None, rul_prediction  # [B, L, D]


