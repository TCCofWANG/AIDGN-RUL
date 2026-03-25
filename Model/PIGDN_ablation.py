# aigcn_pde_phys.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================
# Physics-Guided Diffusion Layer (extends your DiffusionLayer)
# =====================================================
class DiffusionLayer(nn.Module):
    def __init__(self, node_count, out_channels, dt=1.0, use_phys_prior=True):
        super(DiffusionLayer, self).__init__()
        self.node_count = node_count
        self.dt = dt
        self.use_phys_prior = use_phys_prior
        self._kappa = nn.Parameter(torch.tensor(0.1))   # diffusion coefficient
        self._alpha = nn.Parameter(torch.tensor(0.05))  # physics prior strength

        # residual correction network
        self.r_phi = nn.Sequential(
            nn.Linear(1, 16), nn.ELU(), nn.Linear(16, 1)
        )
        self.proj = nn.Conv2d(1, out_channels, kernel_size=(1, 1), bias=True)

    def forward(self, x, A, phys_prior=None):
        """
        x: (B, F, C, M)
        A: (B, M, C, C)
        phys_prior: optional (B, C, M)
        """
        B, F_dim, C, M = x.shape
        device = x.device
        s = x.mean(dim=1)  # (B, C, M)

        # --- Graph Laplacian ---
        deg = A.sum(dim=-1)  # (B, M, C)
        s_for_mul = s.permute(0, 2, 1).unsqueeze(-1)  # (B, M, C, 1)
        As = torch.einsum('bmcd,bmcn->bmdn', (A, s_for_mul)).squeeze(-1)  # (B, M, C)
        Ls = (deg * s.permute(0, 2, 1) - As).permute(0, 2, 1)  # (B, C, M)

        kappa = F.softplus(self._kappa)

        # --- Nonlinear residual term ---
        s_for_r = s.mean(dim=-1, keepdim=True)  # (B, C, 1)
        r_in = s_for_r.view(-1, 1)
        r_out = self.r_phi(r_in).view(B, C).unsqueeze(-1).expand(-1, -1, M)

        # --- Physics-guided correction ---
        if self.use_phys_prior and phys_prior is not None:
            phys_corr = self._alpha * phys_prior  # scaled contribution
        else:
            phys_corr = 0.0

        # Euler update
        s_new = s - self.dt * kappa * Ls + self.dt * (r_out + phys_corr)

        proj_in = s_new.unsqueeze(1)
        corr = self.proj(proj_in)  # (B, out_channels, C, M)
        return corr


# =====================================================
# Basic nconv / linear utilities
# =====================================================
class nconv(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x, A):
        return torch.einsum('bfnm,bmnv->bfvm', (x, A)).contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1))
    def forward(self, x): return self.mlp(x)


# =====================================================
# GCN layer with integrated diffusion-based physics correction
# =====================================================
class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nconv()
        self.mlp = linear((order * support_len + 1) * c_in, c_out)
        self.dropout = dropout
        self.order = order
        # self.diffusion = None
        self._c_out = c_out

    def forward(self, x, bases, phys_prior=None):
        B, F, C, M = x.shape
        out = [x]
        for b in bases:
            x1 = self.nconv(x, b)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, b)
                out.append(x2)
                x1 = x2

        h = self.mlp(torch.cat(out, dim=1))
        # if self.diffusion is None:
        #     self.diffusion = DiffusionLayer(node_count=C, out_channels=self._c_out).to(h.device)

        A_for_diffusion = bases[-1] if len(bases) > 0 else torch.eye(C, device=h.device).view(1, 1, C, C).expand(B, M, C, C)
        # diffusion_corr = self.diffusion(x, A_for_diffusion, phys_prior)
        return h


# =====================================================
# Full AIGCN-PDE model with Physics Priors
# =====================================================
class PIGDN_ablation(nn.Module):
    def __init__(self, args, supports=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_num = args.feature_num
        self.sequence_len = args.sequence_len
        self.hidden_dim = args.hidden_dim
        self.fc_layer_dim = args.d_ff
        self.dropout = nn.Dropout(args.fc_dropout)
        self.AATE_dim = args.AATE_dim
        self.M = 1
        self.supports = supports if supports is not None else []
        self.supports_len = len(self.supports) + 1
        self.n_layer = args.nlayer
        self.outlayer = args.outlayer

        # optional Physics Encoder
        self.use_phys_encoder = getattr(args, "use_phys_encoder", True)
        if self.use_phys_encoder:
            self.phys_encoder = nn.Sequential(
                nn.Conv1d(self.feature_num, self.hidden_dim, kernel_size=3, padding=1),
                nn.ELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(self.hidden_dim, self.feature_num)
            )

        # Aging-guided gates
        self.ll1, self.ll2, self.m_gate1, self.m_gate2 = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.n_layer):
            self.ll1.append(nn.Linear(args.sequence_len, self.AATE_dim))
            self.ll2.append(nn.Linear(args.sequence_len, self.AATE_dim))
            self.m_gate1.append(nn.Sequential(nn.Linear(args.sequence_len + self.AATE_dim, 1), nn.ELU()))
            self.m_gate2.append(nn.Sequential(nn.Linear(args.sequence_len + self.AATE_dim, 1), nn.ELU()))

        # Graph + output layers
        self.gconv = nn.ModuleList([gcn(args.sequence_len, self.hidden_dim, self.dropout,
                                        support_len=self.supports_len, order=args.hop)
                                    for _ in range(self.n_layer)])
        self.project = nn.Linear(args.sequence_len, self.AATE_dim)
        self.temporal_agg = (nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim))
                             if self.outlayer == "Linear"
                             else nn.Sequential(nn.Conv1d(args.hidden_dim, args.hidden_dim, kernel_size=self.M)))
        self.decoder = nn.Sequential(nn.Linear(args.hidden_dim, args.fc_layer_dim), nn.ELU(), nn.Linear(args.fc_layer_dim, 1))
        self.channel_decoder = nn.Sequential(nn.Linear(args.feature_num, args.feature_fc_layer_dim), nn.ELU(),
                                             nn.Linear(args.feature_fc_layer_dim, 1))

    def forward(self, x, OCC=None):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, self.M, -1)  # (B, C, M, N)
        if OCC is None:
            OCC = torch.zeros(B, self.sequence_len, C, device=self.device)
        OCC = self.project(OCC.transpose(1, 2).to(self.device))  # (B, C, AATE_dim)
        OCC = OCC.repeat(self.M, 1, C, 1).permute(1, 0, 2, 3)

        # optional physical prior
        # phys_prior = None
        # if self.use_phys_encoder:
        #     phys_prior = self.phys_encoder(x.squeeze(2))  # (B, C)
        #     phys_prior = phys_prior.unsqueeze(-1).expand(-1, -1, self.M)  # (B, C, M)

        for layer in range(self.n_layer):
            if layer > 0: x_last = x.clone()
            AATE = OCC.view(B, self.M, C, self.AATE_dim)
            AATE_T = OCC.view(B, self.M, self.AATE_dim, C)

            m1 = self.m_gate1[layer](torch.cat([x, AATE.permute(0, 2, 1, 3)], dim=-1))
            m2 = self.m_gate2[layer](torch.cat([x, AATE_T.permute(0, 3, 1, 2)], dim=-1))
            e1 = self.dropout(F.softmax(F.elu_(m1 * self.ll1[layer](x)), dim=-1))
            e2 = self.dropout(F.softmax(F.elu_(m2 * self.ll2[layer](x)), dim=-1))
            e1 = AATE + e1.permute(0, 2, 1, 3)
            e2 = AATE_T + e2.permute(0, 2, 3, 1)
            adp = F.softmax(F.elu_(torch.matmul(e1, e2)), dim=-1)
            new_supports = self.supports + [adp]

            x = self.gconv[layer](x.permute(0, 3, 1, 2), new_supports)
            x = F.elu_(self.dropout(x)).permute(0, 2, 3, 1)
            if layer > 0: x = x + x_last

        x = x.mean(dim=2) if self.outlayer == "Linear" else x.view(B * C, self.M, -1).permute(0, 2, 1)
        if self.outlayer == "CNN":
            x = self.temporal_agg(x).view(B, C, -1)
        else:
            x = self.temporal_agg(x)

        fc_output = self.dropout(x)
        decoder_output = self.decoder(fc_output)
        cd_output = self.channel_decoder(decoder_output.permute(0, 2, 1))
        rul_prediction = torch.abs(cd_output.squeeze(-1))
        return None, rul_prediction
