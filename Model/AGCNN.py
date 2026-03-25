import torch
from torch import nn
from layers.AGCNN_base import *

class AGCNN(nn.Module):
    def __init__(self, input_len, num_features, m, rnn_hidden_size, dropout_rate, bidirectional, fcn_hidden_size):
        super(AGCNN, self).__init__()
        self.fw = Feature_Weighting(num_feature=num_features, m=m)
        self.bigru = BGRU(num_feature=num_features, hidden_size=rnn_hidden_size, dropout_rate=dropout_rate, bidirectional=bidirectional)
        self.cnn = CNN(input_len=input_len)
        D = 2 if bidirectional else 1
        self.fcn = FCN(input_len=input_len, hidden_size=fcn_hidden_size, dropout_rate=dropout_rate, rnn_final_hidden_size=rnn_hidden_size[-1]*D)

    def forward(self, X, idx_x=None):
        X = self.fw(X)
        X = self.bigru(X)
        X = self.cnn(X)
        X = self.fcn(X)
        return None, X
