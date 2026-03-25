import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLayer ( nn.Module ) :
    def __init__(self , node_count , out_channels , use_dynamic_encode=True , kappa=0.1 , alpha=0.05) :
        super ( DiffusionLayer , self ).__init__ ( )
        self.node_count = node_count
        # self.dt = dt
        self.use_dynamic_encode = use_dynamic_encode
        self._kappa = nn.Parameter ( torch.tensor ( kappa ) )  # diffusion coefficient
        self._alpha = nn.Parameter ( torch.tensor ( alpha ) )  # dynamic prior strength

        # residual correction network
        self.r_phi = nn.Sequential ( nn.Linear ( 1 , 16 ) , nn.ELU ( ) , nn.Linear ( 16 , 1 ) )
        self.proj = nn.Conv2d ( 1 , out_channels , kernel_size = (1 , 1) , bias = True )

    def forward(self , x , A , dyn_porior=None) :
        """
        sensor data x: (B, F, C, M)
        adjacency A: (B, M, C, C)
        dyn_corr: (B, C, M)
        """
        B , F_dim , C , M = x.shape
        device = x.device
        s = x.mean ( dim = 1 )  # (B, C, M)

        # --- Graph Laplacian ---
        deg = A.sum ( dim = -1 )  # (B, M, C)
        s_for_mul = s.permute ( 0 , 2 , 1 ).unsqueeze ( -1 )  # (B, M, C, 1)
        As = torch.einsum ( 'bmcd,bmcn->bmdn' , (A , s_for_mul) ).squeeze ( -1 )  # (B, M, C)
        Ls = (deg * s.permute ( 0 , 2 , 1 ) - As).permute ( 0 , 2 , 1 )  # (B, C, M)

        kappa = F.softplus ( self._kappa )

        # --- Nonlinear residual term ---
        s_for_r = s.mean ( dim = -1 , keepdim = True )  # (B, C, 1)
        r_in = s_for_r.view ( -1 , 1 ) #(B*C, 1)
        r_out = self.r_phi ( r_in ).view ( B , C ).unsqueeze ( -1 ).expand ( -1 , -1 , M )# (B, C, M)

        # --- degradation-guided correction ---
        if self.use_dynamic_encode and dyn_porior is not None :
            dyn_corr = self._alpha * dyn_porior  # (B, C, M)  scaled contribution
        else :
            dyn_corr = 0.0

        # Euler update
        s_new = s - kappa * Ls +  (r_out + dyn_corr) # (B, C, M)

        proj_in = s_new.unsqueeze ( 1 ) # (B, 1, C, M)
        corr = self.proj ( proj_in )  # (B, out_channels, C, M)
        return corr

class ACE ( nn.Module ) :
    def __init__(self , layers , sequence_len , AATE_dim ,dropout, single_path=False) :
        super ( ).__init__ ( )
        self.device = torch.device ( "cuda" if torch.cuda.is_available ( ) else "cpu" )
        self.sequence_len = sequence_len
        self.AATE_dim = AATE_dim
        self.single_path = single_path
        self.n_layer = layers
        self.dropout = dropout

        # Aging-guided gates
        self.ll1 , self.ll2 , self.m_gate1 , self.m_gate2 = nn.ModuleList ( ) , nn.ModuleList ( ) , nn.ModuleList ( ) , nn.ModuleList ( )
        for _ in range ( self.n_layer ) :
            self.ll1.append ( nn.Linear ( sequence_len , self.AATE_dim ) )
            self.ll2.append ( nn.Linear ( sequence_len , self.AATE_dim ) )
            self.m_gate1.append ( nn.Sequential ( nn.Linear ( sequence_len + self.AATE_dim , 1 ) , nn.Sigmoid ( ) ) )
            self.m_gate2.append ( nn.Sequential ( nn.Linear ( sequence_len + self.AATE_dim , 1 ) , nn.Sigmoid ( ) ) )

    def forward(self , layer , x , AATE , AATE_T) :
        m = self.m_gate1[layer] ( torch.cat ( [x , AATE.permute ( 0 , 2 , 1 , 3 )] , dim = -1 ) )  # (B, C, M, M)
        e = self.dropout ( F.softmax ( F.elu_ ( m * self.ll1[layer] ( x ) ) , dim = -1 ) )  # (B, C, M, AATE_dim)
        e = AATE + e.permute ( 0 , 2 , 1 , 3 )  # (B, M, C, AATE_dim)

        if not self.single_path :
            mT = self.m_gate2[layer] ( torch.cat ( [x , AATE_T.permute ( 0 , 3 , 1 , 2 )] , dim = -1 ) )  # (B, C, M, M)
            eT = self.dropout ( F.softmax ( F.elu_ ( mT * self.ll2[layer] ( x ) / np.sqrt(self.sequence_len)) , dim = -1 ) )  # (B, C, M, AATE_dim)
            eT = AATE_T + eT.permute ( 0 , 2 , 3 , 1 )  # (B, M, AATE_dim, C)
        else:
            eT = e.permute ( 0 , 1 , 2 , 3 )

        adj = torch.matmul(e, eT)
        adj = F.softmax(F.elu_(adj), dim=-1)

        adj = 0.9 * adj + 0.1 * torch.eye(adj.size(-1), device=adj.device)
        return adj #F.softmax ( F.elu_ ( torch.matmul ( e , eT ) ) , dim = -1 )  # (B, M, C, C)


class nconv ( nn.Module ) :
    def __init__(self) : super ( ).__init__ ( )

    def forward(self , x , A) :
        return torch.einsum ( 'bfnm,bmnv->bfvm' , (x , A) ).contiguous ( )


class linear ( nn.Module ) :
    def __init__(self , c_in , c_out) :
        super ( ).__init__ ( )
        self.mlp = nn.Conv2d ( c_in , c_out , kernel_size = (1 , 1) )

    def forward(self , x) : return self.mlp ( x )


class gcn ( nn.Module ) :
    def __init__(self , c_in , c_out , dropout , feature_num , support_len=3 , order=2) :
        super ( ).__init__ ( )
        self.nconv = nconv ( )
        self.mlp = linear ( (order * support_len + 1) * c_in , c_out )
        self.dropout = dropout
        self.order = order
        self.diffusion = None
        self._c_out = c_out

    def forward(self , x , supports , dyn_corr=None , kappa=None , alpha=None) :
        B , F , C , M = x.shape
        out = [x]
        for b in supports :
            x1 = self.nconv ( x , b )
            out.append ( x1 )
            for k in range ( 2 , self.order + 1 ) :
                x2 = self.nconv ( x1 , b )
                out.append ( x2 )
                x1 = x2

        h = self.mlp ( torch.cat ( out , dim = 1 ) )
        if dyn_corr is not None :
            use_dynamic_encode = True
        else :
            use_dynamic_encode = False

        if self.diffusion is None :
            self.diffusion = DiffusionLayer ( node_count = C , out_channels = self._c_out ,
                                              use_dynamic_encode = use_dynamic_encode , kappa = kappa ,
                                              alpha = alpha ).to ( h.device )

        A_for_diffusion = supports[-1] if len ( supports ) > 0 else torch.eye ( C , device = h.device ).view ( 1 , 1 , C , C ).expand ( B , M , C , C )
        diffusion_corr = self.diffusion ( x , A_for_diffusion , dyn_corr )

        return h + diffusion_corr


class DPDG_aligned ( nn.Module ) :
    def __init__(self , args , supports=None) :
        super ( ).__init__ ( )
        self.device = torch.device ( "cuda" if torch.cuda.is_available ( ) else "cpu" )
        self.feature_num = args.feature_num
        self.sequence_len = args.sequence_len
        self.hidden_dim = args.hidden_dim
        self.fc_layer_dim = args.d_ff
        self.dropout = nn.Dropout ( args.fc_dropout )
        self.AATE_dim = args.AATE_dim
        self.M = args.M if args.M is not None else 1
        self.supports = supports if supports is not None else []
        self.supports_len = len ( self.supports ) + 1
        self.n_layer = args.nlayer
        self.outlayer = args.outlayer
        self.kappa = args.kappa
        self.alpha = args.alpha

        # optional Dynamic Degradation Encoder
        self.use_dynamic_encoder = getattr ( args , "use_dynamic_encoder" , True )
        if self.use_dynamic_encoder :
            self.dynamic_encoder = nn.Sequential (
                nn.Conv1d ( self.feature_num , self.hidden_dim , kernel_size = 3 , padding = 1 ) , nn.ELU ( ) ,
                nn.Conv1d ( self.hidden_dim , self.feature_num , kernel_size = 1 ) , nn.AdaptiveAvgPool1d ( self.M ) )
        self.project = nn.Linear ( args.sequence_len , self.AATE_dim )
        self.project_hidden_dim = nn.Linear ( args.sequence_len , self.hidden_dim )

        # Aging-context
        self.ACE =ACE(self.n_layer, self.hidden_dim, self.AATE_dim, self.dropout)
        # Graph + output layers
        self.gconv = nn.ModuleList ( [gcn ( args.hidden_dim , self.hidden_dim , self.dropout , self.feature_num ,
                                            support_len = self.supports_len , order = args.hop ) for _ in
                                      range ( self.n_layer )] )

        self.temporal_agg = (nn.Sequential (
            nn.Linear ( args.hidden_dim , args.hidden_dim ) ) if self.outlayer == "Linear" else nn.Sequential (
            nn.Conv1d ( args.hidden_dim , args.hidden_dim , kernel_size = self.M ) ))
        self.decoder = nn.Sequential ( nn.Linear ( args.hidden_dim , args.fc_layer_dim ) , nn.ELU ( ) ,
                                       nn.Linear ( args.fc_layer_dim , 1 ) )
        self.channel_decoder = nn.Sequential ( nn.Linear ( args.feature_num , args.feature_fc_layer_dim ) , nn.ELU ( ) ,
                                               nn.Linear ( args.feature_fc_layer_dim , 1 ) )

    def forward(self , x_in , OCC=None) :
        B , N , C = x_in.shape  # B=128, N=50, C=14, M=2
        x = x_in.permute ( 0 , 2 , 1 ).unsqueeze ( 2 ).expand ( -1 , -1 , self.M , -1 )  # (B, C, M, N)
        x = self.project_hidden_dim(x)
        if OCC is None :  # (B, N, 1)
            OCC = torch.zeros ( B , self.sequence_len , C , device = self.device )
        OCC = self.project ( OCC.transpose ( 1 , 2 ).to ( self.device ) )  # (B, 1, AATE_dim)
        OCC = OCC.repeat ( self.M , 1 , C , 1 ).permute ( 1 , 0 , 2 , 3 )  # (B, M, C, AATE_dim)

        # latent degradation satate
        dyn_corr = None
        adj_list = []
        if self.use_dynamic_encoder :
            dyn_corr = self.dynamic_encoder ( x_in.permute ( 0 , 2 , 1 ) )  # (B, C, M) <- (B, N, C)

        for layer in range ( self.n_layer ) :
            if layer > 0 : x_last = x.clone ( )
            AATE = OCC.view ( B , self.M , C , self.AATE_dim )  # (B, M, C, AATE_dim)
            AATE_T = OCC.view ( B , self.M , self.AATE_dim , C )  # (B, M, AATE_dim, C)
            adj = self.ACE( layer, x, AATE, AATE_T)
            new_supports = self.supports + [adj]
            adj_list.append(adj.mean(dim=1))

            c_out = self.gconv[layer] ( x.permute ( 0 , 3 , 1 , 2 ) , new_supports , dyn_corr , self.kappa ,  self.alpha )  # (B, H, C, M)
            c_out = F.elu_ ( self.dropout ( c_out ) ).permute ( 0 , 2 , 3 , 1 )  # (B, C, M, H)
            if layer > 0 : x = c_out + x_last
            else : x = c_out

        adj_stack = torch.stack(adj_list, dim=1)
        c_out = c_out.mean ( dim = 2 ) if self.outlayer == "Linear" else x.view ( B * C , self.M , -1 ).permute ( 0 , 2 , 1 )  # (B, C, H)
        if self.outlayer == "CNN" :
            c_out = self.temporal_agg ( c_out ).view ( B , C , -1 )
        else :
            c_out = self.temporal_agg ( c_out )

        fc_output = self.dropout ( c_out )
        decoder_output = self.decoder ( fc_output )  # (B, C, 1)
        cd_output = self.channel_decoder ( decoder_output.permute ( 0 , 2 , 1 ) )  # (B, 1, 1)
        rul_prediction = torch.abs ( cd_output.squeeze ( -1 ) )  # (B, 1(RUL))
        return adj_stack , rul_prediction
