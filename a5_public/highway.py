#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch as nn
import torch.nn.functional as F

class Highway(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.l_proj = nn.Linear(dim, dim)
        self.l_gate = nn.Linear(dim, dim)

    def forward(self, x_conv_out):
        x_proj = self.l_proj(x_conv_out)
        x_proj = F.relu(x_proj)
        x_gate = self.l_gate(x_conv_out)
        x_gate = F.sigmoid(x_gate)

        x_highway = x_proj * x_gate + x_conv_out * (1-x_gate)

        return x_highway
### END YOUR CODE

