import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class Autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs,input_feature):
        super(Autoformer, self).__init__()

        self.seq_len = configs.input_length
        self.pred_len = configs.input_length

        self.pred_dim = 1
        self.output_attention = False
        self.input_dim = input_feature

        # Decomp，Kernel size of the incoming parameter mean filter
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The embedding operation, since time series are naturally sequential in timing, the role of embedding here is more to adjust the dimensionality
        self.enc_embedding = DataEmbedding_wo_pos(input_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(input_feature, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)



        # Encoder，Multi-coded layer stacking is used
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    #Feature dimension setting during encoding
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    #activation function
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            #Time series are usually applied using the Layernorm and not the BN layer
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder is also stacked with multiple decoders.
        self.decoder = Decoder(
            [
                DecoderLayer(
                    # As in the traditional Transformer structure, the first attention of the decoder needs to be masked to ensure that the prediction at the current position cannot see the previous content.
                    #This approach is derived from NLP practice, but in the case of temporal prediction, there should be no need to use the mask mechanism.
                    #As you can see in the subsequent code, none of the attention modules here actually use the mask mechanism.
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    input_feature,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, input_feature, bias=True)
        )
        self.predictor = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(input_feature*self.seq_len, self.pred_dim)

        )



    def forward(self, x_enc, idx_x=None):
        # decomp init
        # Because generative prediction needs to be used, it is necessary to occupy the prediction section with means and zeros.
        seasonal_init, trend_init = self.decomp(x_enc)

        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        dec_out = self.decoder(dec_out, enc_out,
                                                 trend=trend_init)

        pred = self.predictor(dec_out)


        return None,pred