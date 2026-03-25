import torch
import torch.nn as nn
import numpy as np



# x-NN for feature extraction
class xNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(xNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, weight = self.multihead_attn(X, X, X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)


# deep hidden physics network
class DeepHPM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepHPM, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, weight = self.multihead_attn(X, X, X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)


# multilayer perceptron for mapping hidden states to six RUL predictions
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.features = input_dim
        params = torch.ones(6)
        params = torch.full_like(params, 10, requires_grad=True)
        self.params = nn.Parameter(params)
        self.dnn = nn.Sequential(
            nn.Linear(self.features, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 6)
        )

    def forward(self, X):
        x = self.dnn(X)
        x = x * self.params
        return x.sum(dim=1)

class PINN(nn.Module):
    def __init__(self, args, hidden_dim, derivatives_order):
        super(PINN, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.order = derivatives_order
        self.input_dim = 1 + self.hidden_dim * (self.order + 1)

        self.xnn = xNN(self.args.input_feature, self.hidden_dim)
        self.mlp = MLP(self.hidden_dim + 1)
        self.mlp.train()
        self.deepHPM = DeepHPM(self.input_dim, 1)


    def net_u(self, x, t):
        hidden = self.xnn(x)
        hidden.requires_grad_(True)
        return self.mlp(torch.concat([hidden, t], dim=1)), hidden


    def net_f(self, x, t):
        t.requires_grad_(True)
        u, h = self.net_u(x, t)
        u = u.reshape(-1, 1)
        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_h = [u]
        for i in range(self.order):
            u_ = torch.autograd.grad(
                u_h[-1], h,
                grad_outputs=torch.ones_like(u_h[-1]),
                retain_graph=True,
                create_graph=True
            )[0]
            u_h.append(u_)
        deri = h
        for data in u_h:
            deri = torch.concat([deri, data], dim=1)
        f = u_t - self.deepHPM(deri)
        return f


    def forward(self, x, **kwargs):
        idxs = kwargs['idx']
        t = idxs.to(x.device).to(torch.float64).reshape(-1,1)
        u, h = self.net_u(x, t)
        f = self.net_f(x, t)
        return u, h, f
