import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import seaborn as sns

"""Based on Version-9.11"""

class Diff_Pooling(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim,d_model,d_ff,
                 dropout=0.3,pooling='mean',**kwargs):
        super().__init__()
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.pooling = pooling
        self.supports_len = 1
        self.gcn = GraphConvNet(in_dim, out_dim, dropout, support_len=self.supports_len)
        self.num = 0

    def visual_Attention(self, A, save_path):
        # 这一版里面输入的A是[support_explicit,support_implicit]
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        tmp = A[0, :].clone()
        num_nodes = tmp.shape[-1]
        index = torch.arange(num_nodes)
        index = torch.concat((index[0::2], index[1::2]))
        tmp = tmp[index.flatten(), :]
        tmp = tmp[:, index.flatten()]
        plt.figure()
        sns.heatmap(tmp.detach().cpu().numpy(), annot=False, cmap='coolwarm',vmin=-1,vmax=1)
        plt.title('Attention_explicit')
        # plt.legend()
        plt.savefig(save_path)
        plt.close()

    def forward(self,x,d_x,max_len):

        x_max_index = torch.max(torch.abs(d_x), dim=-1,keepdim=True)[1]
        x_max_index_0 = x_max_index + 1
#        x_max_index_1 = x_max_index - 1
#        x_max_index_0 = torch.where(x_max_index_0 >= max_len, x_max_index_1, x_max_index_0)

        pooling_index = torch.concat([x_max_index, x_max_index_0], dim=-1)

        pooling_value = torch.gather(x, dim=-1, index=pooling_index.long())
        B,H,D,_ = pooling_value.shape
        pooling_value = pooling_value.view(B, H, 2*D)

        pooling_adj = torch.matmul(pooling_value.transpose(-1, -2), pooling_value)/pooling_value.shape[-1]
        eyes_like = torch.eye(2*D).repeat(B, 1, 1).cuda()
        eyes_like_inf = eyes_like*1e8
        pooling_adj = torch.softmax(F.leaky_relu(pooling_adj-eyes_like_inf), dim=-1)+eyes_like

        if self.training is False:
            if self.num < 1:
                self.num += 1
                self.visual_Attention(pooling_adj, save_path='./adj.png')

        pooling_value = self.gcn(pooling_value.unsqueeze(-1), [pooling_adj]).squeeze()
        pooling_value = pooling_value.view(B, H, D, 2)
        pooling_value = pooling_value.permute(0, 2, 3, 1).contiguous()
        pooling_value = pooling_value.view(B, D, 2*H)
        pooling_value = pooling_value.transpose(-1, -2)
        # if self.pooling == 'mean':
        #     pooling_value = torch.mean(pooling_value, dim=-1)
        # if self.pooling == 'max':
        #     pooling_value = torch.max(pooling_value, dim=-1)
        #pooling_value = pooling_value.squeeze()

        return pooling_value


class TCN_base(nn.Module):
    def __init__(self, num_nodes, in_dim, out_dim,d_model,d_ff,
                 dropout=0.3, kernel_size=2, blocks=3, layers=1,args=None,**kwargs):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.args=args

        residual_channels = d_model
        dilation_channels = d_model
        skip_channels = d_ff
        end_channels = d_ff

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                        out_channels=residual_channels,
                                        kernel_size=(1, 1))

        depth = list(range(blocks * layers))

        self.diff_pooling = ModuleList([Diff_Pooling(num_nodes, dilation_channels, residual_channels,d_model,d_ff) for _ in depth])

        receptive_field = 1

        self.supports_len = 3

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList([Conv2d(dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList([Conv2d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList([GraphConvNet(dilation_channels, residual_channels, dropout, support_len=self.supports_len, add_cross=True)
                                              for _ in depth])

        self.diff_index_graph_norm = ModuleList([nn.BatchNorm1d(residual_channels) for _ in depth])

        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()
        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1 # dilation
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

    def diff_x(self,x):
        d_x = torch.diff(x,n=1,dim=-1) #[B,n-1,D,p]
        i_x = x[:,:,:,1:]
        return i_x,d_x



    def forward(self, x,adj,**kargs):
        # adj = [adj_dynamic_with_explicit,adj_dynamic_with_implicit]
        mode = kargs.get('mode')
        x = self.start_conv(x)

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(x, (self.receptive_field - in_len, 0, 0, 0))
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            # EACH BLOCK
            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |   |-dil_conv -- tanh --|                |
            #         ---|                  * ----|-- 1x1 -- + -->	*x_in*
            #                |-dil_conv -- sigm --|    |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            residual = x
            i_x,d_x = self.diff_x(x)
            pooling_value = self.diff_pooling[i](x,d_x,d_x.shape[-1])
            # dilated convolution
            filter = torch.tanh(self.filter_convs[i](i_x))
            gate = torch.sigmoid(torch.abs(self.gate_convs[i](d_x)))
            x = filter * gate
            # parametrized skip connection
            s = self.skip_convs[i](x)

            d_x_max = torch.max(d_x, dim=-1)[0]
            d_x_min = torch.min(d_x, dim=-1)[0]
            adj_d_x_max = torch.where(d_x_max>-d_x_min, d_x_max, d_x_min)

            g1 = F.softmax(F.relu(torch.bmm(adj_d_x_max.permute(0,2,1), adj_d_x_max)), dim=-1)

            d_x_max_index = torch.max(torch.abs(d_x), dim=-1)[1].type(torch.float64)
            d_x_max_index = self.diff_index_graph_norm[i](d_x_max_index)

            g2 = F.softmax(F.relu(torch.bmm(d_x_max_index.permute(0,2,1), d_x_max_index)), dim=-1)

            if len(adj) == 1:
                adj.append(g1)
                adj.append(g2)
            else:
                adj.pop()
                adj.pop()
                adj.append(g1)
                adj.append(g2)


            # what are we skipping??
            try:  # if i > 0 this works
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break

            x = self.graph_convs[i](x, adj, pooling_value)

            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)  # ignore last X?
        B, _, N, _ = x.shape
        x = x.transpose(1,-1)
        return x


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""

    if len(A.shape) == 2:
        return torch.einsum('ncvl,vw->ncwl', (x, A.to(x.device))).contiguous()

    if len(A.shape) == 3:
        return torch.einsum('ncvl,nvw->ncwl', (x, A.to(x.device))).contiguous()


class GraphConvNet(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=1, order=2, add_cross=False):
        super().__init__()
        c_in = (order * support_len + 3) * c_in if add_cross else (order * support_len + 1) * c_in
        self.final_conv = Conv2d(c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list, cross_x=None):
        out = [x]
        for a in support:
            x1 = nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = nconv(x1, a)
                out.append(x2)
                x1 = x2
        if cross_x is not None:
            out.append(cross_x.unsqueeze(-1).repeat(1, 1, 1, x2.shape[-1]))
        h = torch.cat(out, dim=1)
        h = self.final_conv(h)
        h = self.bn(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

