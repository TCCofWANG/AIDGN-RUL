# coding=utf-8
# 单纯的起到降噪作用

import numpy as np
import torch
import torch.nn as nn


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """

    # 随机获取抽取的傅里叶分量的index
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    # FEB
    def __init__(self, in_channels, out_channels, seq_len, modes=0, mode_select_method='random'):
        super(FourierBlock, self).__init__()
        print('fourier enhanced block used!')
        """
        1D Fourier block. It performs representation learning on frequency domain, 
        it does FFT, linear transform, and Inverse FFT.    
        """
        # get modes on frequency domain
        self.index = get_frequency_modes(seq_len, modes=modes, mode_select_method=mode_select_method)
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        # 随机权重
        self.weights1 = nn.Parameter(
            self.scale * torch.ones(in_channels, out_channels, len(self.index), dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # 矩阵乘法
        # (batch, in_channel), (in_channel, out_channel) -> (batch, out_channel)
        return torch.einsum("bh,hi->bi", input, weights)

    def forward(self, q):
        # size = [B, L, D]
        B, L, D = q.shape
        x = q.permute(0, 2, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(B, D, L // 2 + 1, device=x.device, dtype=torch.cfloat)

        # 不使用学习参数


        for wi, i in enumerate(self.index):
            # 随机挑选傅里叶分量
            # 保留下来的傅里叶分量会乘以一个随机权重
            out_ft[:, :,  wi] = self.compl_mul1d(x_ft[:, :,  i], self.weights1[:, :,  wi])
        # Return to time domain
        # x的维度在这个过程中是没有变化的
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

