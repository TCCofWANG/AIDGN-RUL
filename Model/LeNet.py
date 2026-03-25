import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self, input_len, num_features):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(int((input_len//4-2)*(num_features//4-2)*16), 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 1))

    def forward(self, x ,idx_x=None):
        b, l, c = x.shape
        x = x.reshape(b, 1, l, c)
        return None,self.net(x)

