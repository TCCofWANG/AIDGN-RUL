import numpy as np
import torch
from torch import nn

import torch.nn.functional as F

class AIGCN_Cos( nn.Module ) :
    def __init__(self, args, supports=None) :
        super(AIGCN_Cos, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_num = args.feature_num
        self.sequence_len = args.sequence_len
        self.hidden_dim = args.hidden_dim
        self.fc_layer_dim = args.fc_layer_dim
        self.dropout = nn.Dropout( args.fc_dropout )
        args.reshape = True

        self.supports_len = 0
        if supports is not None :
            self.supports_len += len( supports )

        self.C = args.feature_num
        self.M = 1
        self.supports = supports
        self.n_layer = args.nlayer
        self.AATE_dim = args.AATE_dim

        if supports is None :
            self.supports = []

        # self.ll1 = nn.ModuleList()
        # self.ll2 = nn.ModuleList()
        # self.m_gate1 = nn.ModuleList()
        # self.m_gate2 = nn.ModuleList()
        # self.n_layer = args.nlayer
        # for _ in range( self.n_layer ) :
        #     self.ll1.append( nn.Linear( args.sequence_len, self.AATE_dim ))
        #     self.ll2.append( nn.Linear( args.sequence_len, self.AATE_dim ) )
        #     self.m_gate1.append(
        #         nn.Sequential( nn.Linear( args.sequence_len + self.AATE_dim, 1 ), nn.Tanh(),
        #                        nn.ReLU()))
        #     self.m_gate2.append( nn.Sequential( nn.Linear( args.sequence_len + self.AATE_dim, 1 ), nn.Tanh(),
        #                                                  nn.ReLU() ))

        self.supports_len += 1

        self.gconv = nn.ModuleList()  # gragh conv
        for _ in range( self.n_layer ) :
            self.gconv.append( gcn( self.sequence_len, self.hidden_dim, self.dropout, support_len = self.supports_len, order = args.hop ) )
        # self.project = nn.Linear( args.sequence_len, self.AATE_dim )
        ### Encoder output layer 
        self.outlayer = args.outlayer

        if (self.outlayer == "Linear") :
            self.temporal_agg = nn.Sequential( nn.Linear( args.hidden_dim, args.hidden_dim ))

        elif (self.outlayer == "CNN") :
            self.temporal_agg = nn.Sequential( nn.Conv1d( self.hidden_dim, args.hidden_dim, kernel_size = self.M ) )

        ### Predictor ###
        self.decoder = nn.Sequential( nn.Linear( args.hidden_dim, args.fc_layer_dim ),
                                      nn.ReLU(),
                                      # nn.Linear( args.fc_layer_dim, args.fc_layer_dim ), ACTIVATION_MAP[self.fc_activation]()( inplace = True ),
                                      nn.Linear( args.fc_layer_dim, 1 ) )

        self.channel_decoder = nn.Sequential( nn.Linear( args.feature_num, args.feature_fc_layer_dim ),
                                              nn.ReLU(), nn.Linear( args.feature_fc_layer_dim, 1 ) )

    def forward(self, x, OCC=None) :

        x = x.permute( 0, 2, 1 )  # self.dropout( fused_out )
        B, C, N = x.shape
        x = x.unsqueeze( 2 ).expand( -1, -1, self.M, -1 )#.permute( 0, 2, 3, 1 )

        # OCC = OCC.transpose(1, 2).to( self.device ).to(torch.float64)
        # OCC = self.project(  OCC )
        # OCC = OCC.repeat( self.M, 1, C, 1 ).permute( 0, 1, 2, 3 )  # .repeat(1, M, 1, self.AATE_dim)
        # new
        # AATE = self.basis_reshaper(  AATE.to( self.device ).to(torch.float64) ) #new---
        # AATE = AATE.transpose(1, 2)#.to( self.device ).to(torch.float64)
        # AATE = self.project(  AATE )
        # AATE =  AATE.unsqueeze( 1 ).expand( -1, self.M, -1, -1 )#new---

        for layer in range( self.n_layer ) :
            ### Aging-gauided graph structure learning ###
            if(layer > 0): # residual
                x_last = x.clone()
            # AATE = OCC.view( B, self.M, C, self.AATE_dim )  #( B, M, C, L_b )
            # AATE_T = OCC.view( B, self.M, self.AATE_dim, C )  #( B, M, L_b, C )
            #
            # m1 = self.m_gate1[layer]( torch.cat( [x, AATE.permute( 0, 2, 1, 3 )], dim = -1 ) ) # (B, C, M, L_bt)
            # m2 = self.m_gate2[layer]( torch.cat( [x, AATE_T.permute( 0, 3, 1, 2 )], dim = -1 ) ) # (B, C, M, L_bt)
            #
            # e1 = self.dropout(F.softmax( nn.ReLU()( m1 * self.ll1[layer]( x ) ), dim = -1 ))  # (B, C, M, L_b)
            # e2 = self.dropout(F.softmax( nn.ReLU()( m2 * self.ll2[layer]( x ) ) , dim = -1 )) # (B, C, M, L_b)
            #
            #
            # e1 = AATE + e1.permute( 0, 2, 1, 3 )  # (B, M, C, L_B)
            # e2 = AATE_T + e2.permute( 0, 2, 3, 1 )  # (B, M, L_B, C)

            # adp = F.softmax( nn.ReLU()( torch.matmul( x.permute( 0, 2, 1, 3 ), x.permute( 0, 2, 3, 1 ) ) ), dim = -1 )  # (B, M, C, C) used 32 1 16 16
            # Normalize along the last dimension (features)
            x_norm = F.normalize(x, p=2, dim=-1)  # (B, M, C, D)

            # Cosine similarity: (B, M, C, C)
            adp = torch.matmul(x_norm.permute(0, 2, 1, 3), x_norm.permute(0, 2, 3, 1))

            # Softmax along last dimension
            adp = F.softmax(adp, dim=-1)
            # adp = self.dropout(adp)
            new_supports = self.supports + [adp]

            x = self.gconv[layer]( x.permute( 0, 3, 1, 2 ), new_supports )  # (B, F, C, M)
            x = nn.ReLU()( self.dropout( x ) ).permute( 0, 2, 3, 1 )  # (B, C, M, F)

            if(layer > 0): # residual addition
                x = x_last + x

        if (self.outlayer == "CNN") :
            x = x.reshape( B * C, self.M, -1 ).permute( 0, 2, 1 )  # (B*C, F, M)
            x = self.temporal_agg( x )  # (B*C, F, M) -> (B*C, F, 1)
            x = x.view( B, C, -1 )  # (B, C, F)

        elif (self.outlayer == "Linear") :
            x = x.mean( dim = 2 )  # x = x.reshape(B, C, -1) # (B, C, M*F)
            x = self.temporal_agg( x )  # (B, C, hid_dim)

        fc_output = self.dropout( x )
        decoder_output = self.decoder( fc_output )
        decoder_output_p = decoder_output.permute( 0, 2, 1 )
        cd_output = self.channel_decoder( decoder_output_p ).permute( 0, 2, 1 )

        rul_prediction = torch.abs(cd_output.squeeze(-1)) #torch.abs( cd_output[:, -1, :] )

        return None, rul_prediction


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
