import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,IEncoderLayer,IEncoder
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.RevIN import RevIN
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from layers.Embed import PositionalEmbedding
class Transformer_domain(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs,input_feature):
        super(Transformer_domain, self).__init__()
        self.seq_len = configs.input_length
        self.pred_rul = 1
        self.output_attention = False
        self.attention_used_time=None
        self.revin = True
        self.nhead = 2
        self.dropout = 0.1
        self.nlayers = 2

        self.channels = input_feature
        self.timeEmbedding = DataEmbedding(self.channels,d_model=configs.d_model)

        encoder_layers = TransformerEncoderLayer(configs.d_model, self.nhead, 512, dropout=self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        self.dropout = nn.Dropout(self.dropout)
        self.decoder = nn.Linear(configs.d_model, self.pred_rul)


    def forward(self, x_enc, mode):

        src = self.timeEmbedding(x_enc)
        output1 = self.transformer_encoder(src)   # 先不加mask
        output1 = self.dropout(output1)
        output2 = self.decoder(output1)[:,-1:,0]

        return output1,output2 #[B,L]


class Discriminator(nn.Module):  # D_y
    def __init__(self, in_features=14) -> None:
        super().__init__()
        self.in_features = in_features
        self.li = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: Tensor, shape [bts, in_features]
        """
        x = ReverseLayer.apply(x, 1)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.in_features).cuda()
            x = torch.cat((x, pad), 0)
            y = self.li(x)[0].unsqueeze(0)
            return y
        return self.li(x)


class ReverseLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class backboneDiscriminator(nn.Module):  # D_f
    def __init__(self, seq_len, d) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.li1 = nn.Linear(d, 1)
        self.li2 = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = ReverseLayer.apply(x, 1)
        out1 = self.li1(x).squeeze(2)
        if x.size(0) == 1:
            pad = torch.zeros(1, self.seq_len).cuda()
            out1 = torch.cat((out1, pad), 0)
            out2 = self.li2(out1)[0].unsqueeze(0)
            return out2
        out2 = self.li2(out1)
        return out2








































