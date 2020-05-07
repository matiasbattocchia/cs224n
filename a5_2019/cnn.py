#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    kernel_size = 5

    def __init__(self, L, e):
        """
        Parameters
        ----------
        L : int
            Maximum word length.
        e : int
            Character embedding dimension.
        """
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=e,
            out_channels=e,
            kernel_size=self.kernel_size
        )

        L_out=(L+2*self.conv.padding[0]-self.conv.dilation[0]*(self.conv.kernel_size[0]-1)-1)//self.conv.stride[0]+1
        print(L_out)
        self.max_pool = nn.MaxPool1d(kernel_size=L_out)

    def forward(self, x_reshaped):
        x_conv_out = self.conv(x_reshaped)
        x_conv_out = F.relu(x_conv_out)
        x_conv_out = self.max_pool(x_conv_out)

        return x_conv_out
### END YOUR CODE

