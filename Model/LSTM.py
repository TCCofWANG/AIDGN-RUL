import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, input_feature):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_feature, input_feature*2, batch_first=True, proj_size=1)

    def forward(self, x, idx_x=None):
        b, l, c = x.shape
        init_h = torch.zeros((1, b, 1), dtype=x.dtype, device=x.device)
        init_c = torch.zeros((1, b, c*2), dtype=x.dtype, device=x.device)
        output, (hn, cn) = self.lstm(x, (init_h, init_c))

        return None, hn.squeeze(0)