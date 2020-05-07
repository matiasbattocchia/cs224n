#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, e_word):
        super().__init__()
        self.l_proj = nn.Linear(e_word, e_word)
        self.l_gate = nn.Linear(e_word, e_word)

    def forward(self, x_conv_out):
        "x_conv_out and x_highway are [word_batch_size, e_word]"
        x_proj = self.l_proj(x_conv_out)
        x_proj = F.relu(x_proj)
        x_gate = self.l_gate(x_conv_out)
        x_gate = torch.sigmoid(x_gate)

        x_highway = x_proj * x_gate + x_conv_out * (1-x_gate)

        return x_highway

    ### END YOUR CODE

if __name__ == '__main__':
    word_batch_size = 10
    e_word = 4

    h = Highway(e_word)
    i = torch.rand(word_batch_size, e_word)
    o = h(i)

    assert o.shape == (word_batch_size, e_word)
    assert not i.equal(o)
    print('Test passed!')
