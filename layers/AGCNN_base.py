import torch
from torch import nn
from copy import deepcopy

class Feature_Weighting(nn.Module):
    def __init__(self, num_feature, m):
        super(Feature_Weighting, self).__init__()
        self.W1 = nn.Parameter(torch.randn(num_feature, m))
        self.W2 = nn.Parameter(torch.randn(m, num_feature))
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, ori_input):
        attn = self.tanh(ori_input @ self.W1)
        norm_attn = self.softmax(attn @ self.W2)
        return ori_input * norm_attn

class BGRU(nn.Module):
    def __init__(self, num_feature, hidden_size:list, dropout_rate, bidirectional):
        super(BGRU, self).__init__()
        self.bigru = nn.ModuleList()
        self.hidden_size = hidden_size
        assert len(hidden_size)>=1
        self.D = 2 if bidirectional else 1
        for i in range(len(hidden_size)):
            hid = hidden_size[i]
            gru = nn.GRU(input_size=num_feature, hidden_size=hid, num_layers=1, batch_first=True, bidirectional=bidirectional)
            torch.nn.init.xavier_uniform_(gru.weight_ih_l0)
            torch.nn.init.xavier_uniform_(gru.weight_hh_l0)
            gru.bias_ih_l0 = torch.nn.Parameter(torch.zeros(int(hid * 3)))
            gru.bias_hh_l0 = torch.nn.Parameter(torch.zeros(int(hid * 3)))
            self.bigru.append(gru)
            num_feature = int(deepcopy(hid) * self.D)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features):
        for i, gru in enumerate(self.bigru):
            init_hidden = torch.zeros(self.D, features.shape[0], self.hidden_size[i], dtype=features.dtype).to(features.device)
            output, _ = gru(features, init_hidden)
            features = output

        return self.dropout(features)

class CNN(nn.Module):
    def __init__(self, input_len):
        super(CNN, self).__init__()
        cnn = [nn.Conv2d(1, 10, kernel_size=(12, 1), stride=(2, 1), padding=(int(input_len/2+5), 0)),
               nn.Tanh(),
               nn.MaxPool2d(kernel_size=(2, 1)),
               nn.Conv2d(10, 14, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
               nn.Tanh(),
               nn.MaxPool2d(kernel_size=(2, 1)),
               nn.Conv2d(14, 1, kernel_size=(1, 1)),
               nn.Tanh(),
               nn.Flatten(),
               ]
        channels = [10, 14, 1]
        i = 0
        for index in range(len(cnn)):
            if isinstance(cnn[index], nn.Conv2d):
                torch.nn.init.xavier_uniform_(cnn[index].weight)
                cnn[index].bias = torch.nn.Parameter(torch.zeros(channels[i]))
                i += 1
        self.cnn = nn.Sequential(*cnn)

    def forward(self, features):
        b, l, f = features.shape
        features = features.reshape(b, 1, l, f)
        return self.cnn(features)

class FCN(nn.Module):
    def __init__(self, input_len, hidden_size:list, dropout_rate, rnn_final_hidden_size):
        super(FCN, self).__init__()
        assert len(hidden_size) == 2
        fcn = [nn.Linear(int(input_len//4*rnn_final_hidden_size), hidden_size[0]),
               nn.ReLU(),
               nn.Linear(hidden_size[0], hidden_size[1]),
               nn.ReLU(),
               nn.Dropout(p=dropout_rate),
               nn.Linear(hidden_size[1], 1),
               nn.Identity()]
        hidden_size.append(1)
        i = 0
        for index in range(len(fcn)):
            if isinstance(fcn[index], nn.Linear):
                torch.nn.init.xavier_uniform_(fcn[index].weight)
                fcn[index].bias = torch.nn.Parameter(torch.zeros(hidden_size[i]))
                i += 1
        self.fcn = nn.Sequential(*fcn)

    def forward(self, features):
        return self.fcn(features)

