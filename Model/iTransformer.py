import torch
import torch.nn as nn
import torch.nn.functional as F
from mixture_of_experts import MoE

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs, input_feature):
        super(iTransformer, self).__init__()
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
        self.encoder = Encoder(
            [
                EncoderLayer(
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
        aging = aging.transpose(1, 2).to( self.device ).to(torch.float64)
        aging = self.project(  aging.permute( 0, 2, 1) )
        aging = self.project_d_model(aging.permute( 0, 2, 1))
        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, None) # covariates (e.g timestamp) can be also embedded as tokens
        enc_out = self.dropout(enc_out)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        # enc_out = self.encoder(enc_out, attn_mask=None)

        enc_out = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N
        fc_output = self.linear(enc_out.permute(0, 2, 1))
        rul_prediction = self.sensor_fc(fc_output.squeeze(-1))  # (B, 1)

        return None, rul_prediction  # [B, L, D]





class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_lin = nn.Linear(d_model, d_model, bias=bias)
        self.k_lin = nn.Linear(d_model, d_model, bias=bias)
        self.v_lin = nn.Linear(d_model, d_model, bias=bias)
        self.out_lin = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, q, k, v, mask=None):
        # q: (B, Lq, D)
        # k,v: (B, Lkv, D)
        B, Lq, _ = q.shape
        _, Lkv, _ = k.shape

        Q = self.q_lin(q).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, Lq, d_head)
        K = self.k_lin(k).view(B, Lkv, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, Lkv, d_head)
        V = self.v_lin(v).view(B, Lkv, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, Lkv, d_head)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Lq, Lkv)
        if mask is not None:
            # mask expected (B, 1, Lq, Lkv) or broadcastable
            attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)  # (B, H, Lq, Lkv)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, H, Lq, d_head)
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)  # (B, Lq, D)
        out = self.out_lin(out)
        return out, attn  # out: attended representation, attn: (B, H, Lq, Lkv)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttentionFusionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        use_gating=True,
        gating_mode="learnable_scalar",  # options: learnable_scalar | per_head | concat_mlp | none
    ):
        super().__init__()
        self.sensor2aging_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.aging2sensor_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)

        self.norm_s1 = nn.LayerNorm(d_model)
        self.norm_a1 = nn.LayerNorm(d_model)
        self.norm_s2 = nn.LayerNorm(d_model)
        self.norm_a2 = nn.LayerNorm(d_model)

        self.ffn_s = FeedForward(d_model, d_ff, dropout)
        self.ffn_a = FeedForward(d_model, d_ff, dropout)

        self.use_gating = use_gating
        self.gating_mode = gating_mode

        if use_gating:
            if gating_mode == "learnable_scalar":
                # single scalar gate for each direction
                self.g_s2a = nn.Parameter(torch.zeros(1))
                self.g_a2s = nn.Parameter(torch.zeros(1))
            elif gating_mode == "per_head":
                # one scalar per head (applied after multihead output)
                self.g_s2a = nn.Parameter(torch.zeros(n_heads))
                self.g_a2s = nn.Parameter(torch.zeros(n_heads))
                # we will reshape per-head gates to (1, D) before applying
            elif gating_mode == "concat_mlp":
                # an MLP gate that takes [x, cross_x] and outputs alpha in (0,1)
                self.gate_mlp = nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                    nn.Sigmoid()
                )
            else:
                raise ValueError("Unsupported gating_mode")
        # optional final projection after fusion (keeps dimension)
        self.out_proj_s = nn.Linear(d_model, d_model)
        self.out_proj_a = nn.Linear(d_model, d_model)

    def _apply_gate(self, x, cross, gate_param, mode):
        # x, cross: (B, L, D)
        if not self.use_gating:
            return cross
        if mode == "learnable_scalar":
            alpha = torch.sigmoid(gate_param)  # scalar
            return alpha * cross + (1 - alpha) * x
        if mode == "per_head":
            # gate_param: (H,), expand to (B, L, D)
            H = gate_param.shape[0]
            d_head = x.shape[-1] // H
            g = torch.sigmoid(gate_param).view(1, 1, H, 1)  # (1,1,H,1)
            # split x and cross into heads
            B, L, D = x.shape
            xh = x.view(B, L, H, d_head)
            ch = cross.view(B, L, H, d_head)
            fused = g * ch + (1 - g) * xh
            return fused.view(B, L, D)
        if mode == "concat_mlp":
            cat = torch.cat([x, cross], dim=-1)
            alpha = self.gate_mlp(cat)  # (B, L, D)
            return alpha * cross + (1 - alpha) * x
        return cross

    def forward(self, sensor, aging, sensor_mask=None, aging_mask=None):
        # Pre-norm (iTransformer style)
        s = self.norm_s1(sensor)
        a = self.norm_a1(aging)

        # sensor queries aging (sensor <- attending aging)
        s2a_out, attn_s2a = self.sensor2aging_attn(s, a, a, mask=aging_mask)
        # aging queries sensor (aging <- attending sensor)
        a2s_out, attn_a2s = self.aging2sensor_attn(a, s, s, mask=sensor_mask)

        # gating / fusion
        if self.use_gating and self.gating_mode == "per_head":
            s_fused = self._apply_gate(sensor, s2a_out, self.g_s2a, "per_head")
            a_fused = self._apply_gate(aging, a2s_out, self.g_a2s, "per_head")
        elif self.use_gating and self.gating_mode == "learnable_scalar":
            s_fused = self._apply_gate(sensor, s2a_out, self.g_s2a, "learnable_scalar")
            a_fused = self._apply_gate(aging, a2s_out, self.g_a2s, "learnable_scalar")
        elif self.use_gating and self.gating_mode == "concat_mlp":
            s_fused = self._apply_gate(sensor, s2a_out, None, "concat_mlp")
            a_fused = self._apply_gate(aging, a2s_out, None, "concat_mlp")
        else:
            s_fused = s2a_out
            a_fused = a2s_out

        # residual + out projection
        sensor = sensor + self.out_proj_s(s_fused)
        aging = aging + self.out_proj_a(a_fused)

        # FFN block with post-norm
        sensor = sensor + self.ffn_s(self.norm_s2(sensor))
        aging = aging + self.ffn_a(self.norm_a2(aging))

        return sensor, aging, attn_s2a, attn_a2s

