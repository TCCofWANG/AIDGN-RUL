import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AGCN(nn.Module):
    def __init__(self, args, num_node):
        super(AGCN, self).__init__()
        self.N = num_node
        self.d = 4         # 节点嵌入维度
        self.C = 1
        self.F = args.d_model
        self.dropout = args.dropout

        self.conv_weight = nn.Parameter(torch.randn(self.F, self.F))
        self.dropout_layer = nn.Dropout(self.dropout)

        self.bn = nn.BatchNorm2d(self.F)

    def forward(self, X_GT, A):
        batch_size, T, N, C = X_GT.shape

        A_X = torch.einsum('btnc,nm->btmc', X_GT, A)

        A_X_W = torch.matmul(A_X, self.conv_weight)
        A_X_W = self.bn(A_X_W.transpose(1,3)).transpose(1,3)

        Z_AGCN = F.relu(A_X_W)

        return Z_AGCN



class GatedBTCN(nn.Module):
    def __init__(self, args):
        super(GatedBTCN, self).__init__()
        self.C = 1
        self.F = args.d_model
        self.C_h = self.F // 2
        self.kernel_size = 2

        # 平衡机制：两次步长卷积
        self.balance_conv1 = nn.Conv2d(
            in_channels=self.F,
            out_channels=self.C_h,
            kernel_size=(self.kernel_size, 1),  # 时间维度卷积，空间维度不卷积
            stride=(2, 1),
            padding=0,
            bias=True
        )
        self.balance_conv2 = nn.Conv2d(
            in_channels=self.F,
            out_channels=self.C_h,
            kernel_size=(self.kernel_size, 1),
            stride=(2, 1),
            padding=0,
            bias=True
        )
        self.balance_bn = nn.BatchNorm2d(self.C_h)  # 拼接后通道数：C_h*2

        # 门控TCN
        self.gate_conv1 = nn.Conv2d(
            in_channels=self.C_h,
            out_channels=self.F,
            kernel_size=(self.kernel_size, 1),
            dilation=(2, 1),
            padding=0
        )
        self.gate_conv2 = nn.Conv2d(
            in_channels=self.C_h,
            out_channels=self.F,
            kernel_size=(self.kernel_size, 1),
            dilation=(2, 1),
            padding=0
        )
        self.gate_bn = nn.BatchNorm2d(self.F)

    def forward(self, X_3D):
        batch_size, T, N, C = X_3D.shape

        X = X_3D.permute(0, 3, 1, 2)  # [B, C, T, N]

        # 平衡机制
        h1 = self.balance_conv1(X)  # [B, C_h, T//2, N]
        h2 = self.balance_conv2(X)  # [B, C_h, T//2, N]
        h = torch.cat([h1, h2], dim=2)
        h = self.balance_bn(h)

        # 门控TCN
        gate1 = self.gate_conv1(h)
        gate2 = self.gate_conv2(h)
        X_GT = torch.tanh(gate1) * torch.sigmoid(gate2)
        X_GT = self.gate_bn(X_GT)

        X_GT = X_GT.permute(0, 2, 3, 1)  # [B, T-2, N, F]

        return X_GT


class ASCN(nn.Module):
    def __init__(self, args, num_node):
        super(ASCN, self).__init__()
        self.N = num_node      # 节点数
        self.F = args.d_model      # 输出特征数
        self.d = 4      # 节点嵌入维度

        # 同步卷积权重矩阵
        self.conv_weight = nn.Parameter(torch.randn(self.F, self.F))

        self.bn = nn.BatchNorm1d(self.F)

    def forward(self, chi_3D,A):
        batch_size, T_local, N, C = chi_3D.shape  # T_local=3
        assert T_local == 3, "ASCN输入时间切片长度必须为3"

        chi_reshaped = chi_3D.reshape(batch_size, 3 * N, C)
        A_chi = torch.einsum('bic,ij->bjc', chi_reshaped, A)
        Z_ASCN = torch.matmul(A_chi, self.conv_weight)
        Z_ASCN = self.bn(Z_ASCN.transpose(1,2)).transpose(1,2)
        Z_final = F.relu(Z_ASCN, inplace=False)

        return Z_final


class Local_ST(nn.Module):
    def __init__(self, args, num_node):
        super(Local_ST, self).__init__()
        self.M = 2      # ASCN层数）
        self.N = num_node      # 节点数
        self.F = args.d_model      # 输出特征数

        self.ascn_layers = nn.ModuleList([ASCN(args,num_node) for _ in range(self.M)])

    def forward(self, X_3D,A):

        batch_size, T, N, C = X_3D.shape
        X_windows = X_3D.unfold(dimension=1, size=3, step=1)  # [B, T-2, N, C, 3]
        X_windows = X_windows.permute(0, 1, 4, 2, 3)  # [B, T-2, 3, N, C]
        batch_size_windows, num_windows = X_windows.shape[0], X_windows.shape[1]
        X_windows_flat = X_windows.reshape(-1, 3, N, C)  # [B*(T-2), 3, N, C]

        Z_ASCN_list = []
        current_output = 0
        for ascn in self.ascn_layers:
            # 批量处理所有时间窗口
            current_output = current_output + ascn(X_windows_flat,A)  # [B*(T-2), 3N, F]
            Z_ASCN_list.append(current_output)

        Z_ASCN_stack = torch.stack(Z_ASCN_list, dim=-1)  # [B*(T-2), M, 3N, F]
        Z_pooled,_ = torch.max(Z_ASCN_stack,dim=-1)  # [B*(T-2), 3N, F]
        Z_final = Z_pooled.reshape(batch_size, T - 2, 3 * N, self.F)

        # 裁剪操作 [B, T-2, N, F]
        Z_CR = Z_final[:, :, self.N:2 * self.N, :]

        return Z_CR


class Global_ST(nn.Module):
    def __init__(self, args, num_node):
        super(Global_ST, self).__init__()
        self.gated_btcn = GatedBTCN(args)  # 全局时间特征提取
        self.agcn = AGCN(args, num_node)   # 全局空间特征提取

    def forward(self, X_3D, A):
        # Gated BTCN提取全局时间特征
        X_GT = self.gated_btcn(X_3D)
        # AGCN提取全局空间特征
        Z_AGCN = self.agcn(X_GT,A)

        return Z_AGCN


class SDAGCN(nn.Module):
    def __init__(self, args, input_feature):
        super(SDAGCN, self).__init__()
        self.args = args
        self.num_node = input_feature
        self.C = 1
        self.F = args.d_model
        self.st_blocks = 1
        self.T = args.input_length
        self.input_fc = nn.Linear(self.C, self.F)
        # 堆叠时空块
        self.st_block_list = nn.ModuleList([
            nn.ModuleDict({
                "local_st": Local_ST(args,self.num_node),
                "global_st": Global_ST(args,self.num_node),
                "bn": nn.BatchNorm2d(self.F)
            }) for _ in range(self.st_blocks)
        ])

        self.output_fc1 = nn.Linear(self.F * self.num_node * (self.T - 2*self.st_blocks), self.F)
        self.output_fc2 = nn.Linear(self.F , 1)  # 输出RUL预测值

        # node emb init
        self.node_emb = nn.Parameter(torch.randn(self.num_node, self.F))

    def forward(self, X, idx=None):

        X = X.unsqueeze(-1)
        batch_size, T, N, C = X.shape

        # 输入层转换（全连接层）
        X = self.input_fc(X)  # [B, T, N, F]

        # adp图的构建
        # 生成A_adp
        sim_matrix = torch.matmul(self.node_emb, self.node_emb.T)
        A_adp = F.softmax(F.relu(sim_matrix), dim=1)  # [N, N]
        # 生成I_TC
        I_TC = torch.eye(N, device=A_adp.device)  # [N, N]

        row1 = torch.cat([A_adp, I_TC, I_TC], dim=1)
        row2 = torch.cat([I_TC, A_adp, I_TC], dim=1)
        row3 = torch.cat([I_TC, I_TC, A_adp], dim=1)
        A_stadp = torch.cat([row1, row2, row3], dim=0)

        # 堆叠时空块
        Z_st_list = []
        for st_block in self.st_block_list:
            # Local_ST与Global_ST并行计算
            Z_local = st_block["local_st"](X,A_stadp)
            Z_global = st_block["global_st"](X,A_adp)
            Z_st = Z_local + Z_global
            Z_st = st_block["bn"](Z_st.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            Z_st_list.append(Z_st)
            X = Z_st  # 作为下一个时空块的输入

        X_final = Z_st_list[-1]
        # 展平特征：[B, T-2, N, F] → [B, (T-2)*N*F]
        Z_flat = X_final.reshape(batch_size, -1)
        y_pred = self.output_fc1(Z_flat)
        y_pred = self.output_fc2(y_pred)  # [B, 1]

        # y_pred = torch.clamp(y_pred, 0, 1)

        return None, y_pred
