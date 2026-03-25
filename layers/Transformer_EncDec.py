import torch
import torch.nn as nn
import torch.nn.functional as F
from PatchTST_layers import *


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 对每一个特征维度的所有batch，所有长度统计均值方差进行标准化
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):

        new_x = self.attention(
            x, x, x
        )
        x = x + self.dropout(new_x)

        y = self.norm1(x.permute(0,2,1)).permute(0,2,1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2((x + y).permute(0,2,1)).permute(0,2,1)

class IEncoderLayer(nn.Module):
    def __init__(self, attention_time,attention_dim, d_model, d_feature,d_ff=None, dropout=0.1, activation="relu"):
        super(IEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_time = attention_time
        self.attention_dim = attention_dim
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 对每一个特征维度的所有batch，所有长度统计均值方差进行标准化
        # 设定多少维度，就计算多少维度的特征
        self.norm1 = nn.BatchNorm1d(d_feature)
        self.norm2 = nn.BatchNorm1d(d_feature)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):


        # 变量间考量注意力关系
        x = x.permute(0,2,1)
        new_x_d = self.attention_dim(
            x, x, x
        )
        x = x.permute(0,2,1) + self.dropout(new_x_d.permute(0,2,1))

        y = x = self.norm1(x.permute(0,2,1)).permute(0,2,1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))

        return self.norm2((x + y).permute(0,2,1)).permute(0,2,1)


class Icross_EncoderLayer(nn.Module):
    def __init__(self, attention_time,attention_dim, d_model, d_feature,n_seg,d_ff=None, dropout=0.1, activation="relu"):
        super(Icross_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention_time = attention_time
        self.attention_dim = attention_dim
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        # 对每一个特征维度的所有batch，所有长度统计均值方差进行标准化
        # 设定多少维度，就计算多少维度的特征
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.n_seg = n_seg
        self.router = nn.Parameter(torch.randn(self.n_seg,d_model))



    def forward(self, x, attn_mask=None):

        # 时间上的考虑注意力关系
        B = x.size()[0]
        batch_router = self.router.unsqueeze(0).repeat(B,1,1) # attention 输入[B,L,D] 找L与L的关系
        new_x_buffer = self.attention_time(
            batch_router, x, x
        )  # output [B,n_seg,D]
        # 变量间考量注意力关系
        new_x_receive = self.attention_dim(
            x, new_x_buffer, new_x_buffer
        )
        x = x.permute(0,2,1) + self.dropout(new_x_receive.permute(0,2,1)) #转维度是为了后续卷积

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))

        return self.norm2((x + y)).permute(0,2,1)


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = nn.Sequential(Transpose(1, 2), norm_layer, Transpose(1, 2))

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)


        if self.norm is not None:
            x = self.norm(x)

        return x



class IEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(IEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = nn.Sequential(Transpose(1, 2), norm_layer, Transpose(1, 2))

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x = attn_layer(x, attn_mask=attn_mask)


        if self.norm is not None:
            x = self.norm(x)

        return x



class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.norm2 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.norm3 = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class T2V_DecoderLayer(nn.Module):
    def __init__(self, self_attention,t2v_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(T2V_DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.t2v_attention = t2v_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.norm3 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, x_date, y_date, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
        ))
        x = self.norm1(x.permute(0,2,1)).permute(0,2,1)

        x = self.dropout(self.t2v_attention(
            x, x, x, x_date, y_date
        ))

        y = x = self.norm2(x.permute(0,2,1)).permute(0,2,1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3((x + y).permute(0,2,1)).permute(0,2,1)




class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.Sequential(Transpose(1, 2), norm_layer, Transpose(1, 2))
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x


class T2V_Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(T2V_Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.Sequential(Transpose(1, 2), norm_layer, Transpose(1, 2))
        self.projection = projection

    def forward(self, x, x_date, y_date, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, x_date, y_date, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
