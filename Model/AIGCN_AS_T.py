import math
import torch
from torch import nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer,IEncoderLayer,IEncoder
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F
from torch.distributions import Gamma


class AIGCN_AS_T( nn.Module ) :
    def __init__(self, configs,input_feature):
        super(AIGCN_AS_T, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.feature_num = args.feature_num
        # self.sequence_len = args.sequence_len
        # self.hidden_dim = args.hidden_dim
        # self.fc_layer_dim = args.fc_layer_dim
        # self.dropout = nn.Dropout( args.fc_dropout )
        self.ablation_mode = configs.ablation_mode
        self.seq_len = configs.input_length
        self.pred_rul = 1
        self.output_attention = False
        self.attention_used_time=None
        self.revin = False
        self.dropout = 0.1
        self.channels = input_feature
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
        # self.decoder = Decoder(
        #     [
        #         DecoderLayer(
        #             AttentionLayer(
        #                 M_FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             AttentionLayer(
        #                 M_FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
        #                 configs.d_model, configs.n_heads),
        #             configs.d_model,
        #             configs.d_ff,
        #             dropout=configs.dropout,
        #             activation=configs.activation,
        #         )
        #         for l in range(configs.d_layers)
        #     ],
        #     norm_layer=torch.nn.BatchNorm1d(configs.d_model),
        #     projection=nn.Linear(configs.d_model, input_feature, bias=True)
        # )



        self.project = nn.Linear( 1, configs.feature_num )
        ### Encoder output layer
        self.outlayer = configs.outlayer

        if (self.outlayer == "Linear") :
            self.temporal_agg = nn.Sequential( nn.Linear( configs.d_model, configs.d_model ))

        elif (self.outlayer == "CNN") :
            self.temporal_agg = nn.Sequential( nn.Conv1d( configs.d_model, configs.d_model, kernel_size = self.M ) )

        ### Predictor ###
        self.decoder = nn.Sequential( nn.Linear( configs.d_model, configs.d_model ),
                                      nn.ReLU(),
                                      # nn.Linear( args.fc_layer_dim, args.fc_layer_dim ), ACTIVATION_MAP[self.fc_activation]()( inplace = True ),
                                      nn.Linear( configs.d_model, 1 ) )

        self.channel_decoder = nn.Sequential( nn.Linear( self.seq_len, configs.d_model ),
                                              nn.ReLU(), nn.Linear( configs.d_model, 1 ) )

    def forward(self, sensor_data, OCC=None) :

        # sensor_data = sensor_data.permute( 0, 2, 1 )  # self.dropout( fused_out )
        B, C, N = sensor_data.shape
        # x = sensor_data.unsqueeze( 2 ).expand( -1, -1, self.M, -1 )  # .permute( 0, 2, 3, 1 )

        torch.manual_seed(0)
        # new linear, exponential, weiner_process, gamma_process
        OCC = OCC.transpose(1, 2).to( self.device ).to(torch.float64)
        if self.ablation_mode=="exponential":
            OCC = 1 - torch.exp(-OCC)

        elif self.ablation_mode=="weiner_process":
            mu = 0.5       # For Wiener process
            sigma = 0.1    # For Wiener process
            OCC = mu * OCC - sigma * torch.randn_like(OCC)
        elif self.ablation_mode=="gamma_process":
            alpha = 2.0      # Shape scaling for Gamma process
            beta = 1.0       # Rate for Gamma process
            OCC = gamma_process(alpha, beta, OCC, device=self.device).to(dtype=torch.float64)

        elif self.ablation_mode=="pca":
            OCC = torch_pca_over_channels_per_time(sensor_data)
        elif self.ablation_mode=="feature_mean":
            OCC = sensor_data.mean(dim=1, keepdim=True)

        elif self.ablation_mode=="pe":
            pos_encoding = get_sinusoidal_encoding_1d(N, self.device)  # (1, 1, 30)
            pos_encoding = pos_encoding.to(dtype=torch.float64)  # Explicit cast
            OCC = pos_encoding.expand(B, 1, N)  # (batch, 1, sequence_len)

        OCC = self.project(  OCC.permute( 0, 2, 1 ) )
        #OCC = OCC.repeat( 1, C, 1 )#.permute( 0, 1, 2 )
        x_enc = sensor_data+OCC
        x_enc = self.timeEmbedding(x_enc)
        enc_out = self.encoder(x_enc, attn_mask=None)

        if (self.outlayer == "CNN") :
            fc_output = self.temporal_agg( enc_out )  # (B*C, F, M) -> (B*C, F, 1)
            fc_output = fc_output.view( B, C, -1 )  # (B, C, F)

        elif (self.outlayer == "Linear") :
            fc_output = self.temporal_agg( enc_out )  # (B, C, hid_dim)

        fc_output = self.dropout( fc_output )
        decoder_output = self.decoder( fc_output )
        decoder_output_p = decoder_output.permute( 0, 2, 1 )
        cd_output = self.channel_decoder( decoder_output_p ).permute( 0, 2, 1 )

        rul_prediction = torch.abs(cd_output.squeeze(-1)) #torch.abs( cd_output[:, -1, :] )

        return None, rul_prediction

def torch_pca_over_channels_per_time(x, n_components=1):
    """
    Perform PCA over the channel dimension for each time step (T),
    using the entire batch. Returns shape: (B, k, T)

    Args:
        x: Tensor of shape (B, C, T)
        n_components: number of principal components to retain

    Returns:
        Tensor of shape (B, k, T)
    """
    B, C, T = x.shape
    pca_out = torch.zeros((B, n_components, T), dtype=x.dtype, device=x.device)

    for t in range(T):
        xt = x[:, :, t]  # shape: (B, C) — all batch samples at time t

        # Center the data across the batch
        xt_mean = xt.mean(dim=0, keepdim=True)  # (1, C)
        xt_centered = xt - xt_mean  # (B, C)

        # Perform SVD on (B, C)
        U, S, Vh = torch.linalg.svd(xt_centered, full_matrices=False)  # Vh: (C, C)

        # Take top-k principal directions
        V_k = Vh[:n_components, :]  # (k, C)

        # Project the centered data onto top-k components
        xt_proj = torch.matmul(xt_centered, V_k.T)  # (B, k)

        # Store in output tensor
        pca_out[:, :, t] = xt_proj

    return pca_out  # shape: (B, k, T)

def torch_pca_over_channels(x, n_components=1):
    B, C, T = x.shape
    pca_out = torch.zeros((B, n_components, T), dtype=x.dtype, device=x.device)

    for t in range(T):
        xt = x[:, :, t]  # (B, C)
        xt_mean = xt.mean(dim=0, keepdim=True)
        xt_centered = xt - xt_mean

        U, S, Vh = torch.linalg.svd(xt_centered, full_matrices=False)
        V_k = Vh[:n_components, :]  # (k, C)
        xt_proj = torch.matmul(xt_centered, V_k.T)  # (B, k)
        pca_out[:, :, t] = xt_proj

    return pca_out  # shape: (B, k, T)

def gamma_process(alpha, beta, t, device):
    """
    Generate a Gamma process where the shape parameter is αt and rate is β.

    Args:
        alpha (float): Base shape parameter.
        beta (float): Rate parameter.
        t (torch.Tensor): Time tensor of shape (batch, 1, seq_len).
        device: Device to run on (e.g., "cuda" or "cpu").

    Returns:
        torch.Tensor: Gamma process samples of shape (batch, 1, seq_len).
    """
    # Ensure t is non-negative and float32
    t = t.float().clamp(min=0)  # shape: (128, 1, 30)

    # Sample Gamma(αt, β) for each t
    gamma_dist = Gamma(alpha * t, beta)
    samples = gamma_dist.sample()  # shape: (128, 1, 30)

    return samples.to(device)

def get_sinusoidal_encoding_1d(sequence_len, device):
    position = torch.arange(sequence_len, device=device)  # (sequence_len,)
    # Use a simple scaling if you want variability:
    div_term = 1 / (10000 ** (torch.arange(0, sequence_len, device=device) / sequence_len))
    encoding = torch.sin(position * div_term).to(torch.float32)
    return encoding.unsqueeze(0).unsqueeze(0)  # (1, 1, sequence_len)

def get_fixed_sinusoidal_encoding(sequence_len, device):
    # Force float32 everywhere
    position = torch.arange(sequence_len, device=device, dtype=torch.float64).unsqueeze(1)  # (sequence_len, 1)
    div_term = torch.exp(-math.log(10000.0) * torch.arange(0, 1, dtype=torch.float64, device=device) / sequence_len)  # (1,)

    encoding = torch.sin(position * div_term)  # (sequence_len, 1)
    encoding = encoding.squeeze(1)  # (sequence_len,)
    encoding = encoding.unsqueeze(0).unsqueeze(0)  # (1, 1, sequence_len)
    return encoding.contiguous()

class gcn( nn.Module ) :
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2) :
        super( gcn, self ).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        # c_in = (order*support_len)*c_in
        self.mlp = linear( c_in, c_out )
        self.dropout = dropout
        self.order = order

    def forward(self, x, bases) :
        # x (B, F, C, M)
        # a (B, M, C, C)
        out = [x]  # 32 30 16 1
        for b in bases :
            x1 = self.nconv( x, b )
            out.append( x1 )
            for k in range( 2, self.order + 1 ) :
                x2 = self.nconv( x1, b )
                out.append( x2 )
                x1 = x2

        h = torch.cat( out, dim = 1 )  # concat x and x_conv #32 60 16 1
        h = self.mlp( h )
        return h


class nconv( nn.Module ) :
    def __init__(self) :
        super( nconv, self ).__init__()

    def forward(self, x, A) :
        # x (B, F, C, M)
        # A (B, M, C, C)
        x = torch.einsum( 'bfnm,bmnv->bfvm', (x, A) )  # used
        # print(x.shape)
        return x.contiguous()  # (B, F, C, M)


class linear( nn.Module ) :
    def __init__(self, c_in, c_out) :
        super( linear, self ).__init__()
        # self.mlp = nn.Linear(c_in, c_out)
        self.mlp = torch.nn.Conv2d( c_in, c_out, kernel_size = (1, 1), padding = (0, 0), stride = (1, 1), bias = True )

    def forward(self, x) :
        # x (B, F, C, M)

        # return self.mlp(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.mlp( x )
