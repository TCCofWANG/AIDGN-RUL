import torch
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from collections import OrderedDict

def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum('nvl,nvw->nwl', (x, A.to(x.device))).contiguous()

class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = nn.Conv1d(c_in, c_out, 1, padding=0, stride=1, bias=True)
        self.dropout = dropout
        self.order = order
        self.bn = nn.BatchNorm1d(c_out)

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)
        h = self.final_conv(h.transpose(-1, 1))
        h = self.bn(h).transpose(-1, 1)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class CDSG(nn.Module):
    def __init__(self, args, input_feature):
        super(CDSG, self).__init__()
        # graph_construction_type = args.graph_construction_type
        self.device = torch.device('cuda')
        self.n_sensors = input_feature
        self.idx_n_sensors = self.n_sensors

        self.patch_len = args.patch_size
        self.patch_stride = args.patch_size
        num_patches = (args.input_length - self.patch_len) // self.patch_stride + 1

        self.d_ff = args.d_ff
        self.d_mdoel = args.d_model

        self.x_nonlin_map = nn.Sequential(
            nn.Conv2d(1, self.d_mdoel, kernel_size=(1,1)),
            nn.BatchNorm2d(self.d_mdoel)
        )

        self.gru = nn.GRU(input_size=1, hidden_size=self.d_mdoel, num_layers=1, batch_first=True)
        self.dgl = DGL(self.patch_len, self.d_mdoel, self.d_mdoel, self.n_sensors)

        self.graph_GCN = GraphConvNet(c_in=self.d_mdoel,c_out=self.d_mdoel,dropout=0.3,support_len=1)

        self.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.d_mdoel * self.idx_n_sensors, 2 * self.d_mdoel)),
            ('relu1', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(2 * self.d_mdoel, 2 * self.d_mdoel)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(2 * self.d_mdoel, self.d_mdoel)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc4', nn.Linear(self.d_mdoel, 1)),

        ]))

    def forward(self, X,idx):
        # print(X.size())  [B,L,N]
        B,L,N = X.shape

        input_X = X.transpose(1,2).reshape(B*N,L).unsqueeze(-1)
        gru_hs, gru_last_h = self.gru(input_X)
        dgl_outputs = self.dgl(gru_hs)
        gcn_output = self.graph_GCN(gru_last_h.transpose(0,1).reshape(B,N,-1), [dgl_outputs[:,-1,:,:]])

        # TODO GRU Ablation
#       input_X = self.x_nonlin_map(X.transpose(1,2).unsqueeze(1)).transpose(1,2).reshape(B*N,self.d_mdoel,L).transpose(1,2)
#       dgl_outputs = self.dgl(input_X) # [B,n_p,N,N]
#       gcn_output = self.graph_GCN(input_X[:,-1,:].reshape(B,N,-1), [dgl_outputs[:,-1,:,:]])  #[B,N,D]

        rul_pred = self.fc(torch.flatten(gcn_output,start_dim=1))

        return None,rul_pred




class DGL(nn.Module):
    def __init__(self, patch_len, input_feature, d_model, N):
        super(DGL, self).__init__()

        self.N = N
        self.patch_len = patch_len
        self.stride = patch_len
        self.gru = nn.GRU(input_size=input_feature, hidden_size=d_model, num_layers=1, batch_first=True)
        self.mlp_e = MLP(in_f=d_model, out_f=N)
        self.mlp_m = MLP(in_f=d_model, out_f=N)



    def forward(self, x):
        # x[B*N,L,H]
        x_p = x.unfold(dimension=1, size=self.patch_len, step=self.stride)  #[B,n,D,p]
        bs, n_p, dimension,p_len = x_p.size()
        gru_inputs = torch.mean(x_p,dim=-1)
        gru_outputs_Ss, gru_outputs_last_S = self.gru(gru_inputs)
        hat_Ss = gru_outputs_Ss + gru_outputs_last_S.transpose(0,1)

        output_mlps = F.relu(self.mlp_e(hat_Ss) * F.sigmoid(self.mlp_m(hat_Ss)))
        A_s = output_mlps.reshape(-1,self.N,n_p,self.N).transpose(1,2)


        return A_s



class MLP(nn.Module):
    def __init__(self, in_f, out_f):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_f, 128)  # 输入到隐藏层
        self.fc2 = nn.Linear(128, 64)       # 隐藏层到中间层
        self.fc3 = nn.Linear(64, out_f)         # 中间层到输出层
        self.relu = nn.ReLU()               # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 输入经过第一个全连接层并激活
        x = self.relu(self.fc2(x))  # 经过第二个全连接层并激活
        x = self.fc3(x)             # 经过输出层
        return x


