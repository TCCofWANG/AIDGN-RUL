import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,IEncoderLayer,IEncoder
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from layers.RevIN import RevIN


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs,input_feature):
        super(Transformer, self).__init__()
        self.seq_len = configs.input_length
        self.pred_rul = 1
        self.output_attention = False
        self.attention_used_time=None
        self.revin = False
        self.dropout = 0.1
        self.channels = input_feature

        # Embedding


        self.timeEmbedding = DataEmbedding(self.channels,d_model=configs.d_model)


        self.dropout = nn.Dropout(self.dropout)


        # if self.revin:self.revin_layer = RevIN(configs.d_model, affine=False, subtract_last=False)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        M_FullAttention(configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads,d_keys=configs.d_model,d_values=configs.d_model),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],

            norm_layer=torch.nn.BatchNorm1d(configs.d_model)
        )

        # Decoder_normal
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        M_FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        M_FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.BatchNorm1d(configs.d_model),
            projection=nn.Linear(configs.d_model, input_feature, bias=True)
        )

        self.predictor = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_feature * self.seq_len, self.pred_rul)

        )


        # Decoder_linear
        # self.decoder = nn.Linear(configs.d_model, self.pred_rul)

    def forward(self, x_enc, idxs=None):

        if self.revin:
            #[B,L,D]
            x_enc = self.revin_layer(x_enc, 'norm')


        # enc_out = self.enc_embedding(x_enc) #[B,L,d_model]

        x_enc = self.timeEmbedding(x_enc)
        enc_out = self.encoder(x_enc, attn_mask=None)

        output1 = self.dropout(enc_out)

        # decoder_normal
        output2 = self.decoder(x_enc,output1)

        pred = self.predictor(output2)

        # decoder_linear
        # output2 = self.decoder(output1)

        return None,pred #[B,1]


















