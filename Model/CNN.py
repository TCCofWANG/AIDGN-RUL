# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:26:57 2020

@author: Utkarsh Panara
"""
import torch
import torch.nn as nn
from layers.Transformer_EncDec import IEncoderLayer,IEncoder
from layers.SelfAttention_Family import M_FullAttention, AttentionLayer

class CNN(torch.nn.Module):

    def __init__(self,configs,input_feature):
        super(CNN, self).__init__()

        self.input_feature = input_feature
        self.d_model = configs.d_model
        self.seq_len = configs.input_length
        self.zeropad = torch.nn.ZeroPad2d((0,0,0,9))
        self.conv1 = torch.nn.Conv2d(1, 10, (10,1), 1,0,1)
        self.conv2 = torch.nn.Conv2d(10,10,(10,1),1,0,1)
        self.conv3 = torch.nn.Conv2d(10,10,(10,1),1,0,1)
        self.conv4 = torch.nn.Conv2d(10,10,(10,1),1,0,1)
        self.conv5 = torch.nn.Conv2d(10,1,(3,1),1,(1,0),1)
        self.fc1 = torch.nn.Linear(self.input_feature*self.seq_len,100)
        self.fc2 = torch.nn.Linear(100,1)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)


        self.encoder = IEncoder(
            [
                IEncoderLayer(
                    AttentionLayer(
                        M_FullAttention(configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False), self.seq_len, configs.n_heads,
                        d_keys=self.seq_len, d_values=self.seq_len),
                    AttentionLayer(
                        M_FullAttention(configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False), self.seq_len, configs.n_heads,
                        d_keys=self.seq_len, d_values=self.seq_len),
                    self.seq_len,
                    self.d_model,
                    self.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            # ITransformer的源码是这么对这一维做norm
            norm_layer=torch.nn.BatchNorm1d(self.d_model)
        )

        self.embedding = nn.Linear(self.input_feature,self.d_model)

        self.output_projection = nn.Linear(self.d_model,self.input_feature)
    def do_patching(self,z,stride):
        z = nn.functional.pad(z, (0, stride))
        # real doing patching
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return z


    def weights_init(self,layer):
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out')
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out')
            if layer.bias is not None:
                layer.bias.data.fill_(0.001)

        return None

    def forward(self, input_):
        input_ = input_.unsqueeze(1)
        out = self.zeropad(input_)
        out = self.conv1(out)
        out = torch.sigmoid(out)

        out = self.zeropad(out)
        out = self.conv2(out)
        out = torch.sigmoid(out)

        out = self.zeropad(out)
        out = self.conv3(out)
        out = torch.sigmoid(out)

        out = self.zeropad(out)
        out = self.conv4(out)
        out = torch.sigmoid(out)

        out = self.conv5(out)
        out = out.view(out.size(0),-1)

        out = self.fc1(out)
        out = torch.sigmoid(out)
        out = self.dropout(out)
        pred = self.fc2(out)


        # 这种输出头的方式是简单的展平


        return pred,None






