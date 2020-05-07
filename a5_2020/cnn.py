#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    kernel_size = 5
    padding = 1

    def __init__(self, e_char, filters):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=e_char,
            out_channels=filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )

    def forward(self, x_reshaped):
        """
        x_reshaped is [word_batch_size, e_char, m_word]
        x_conv is [word_batch_size, filters, m_word-kernel_size+1]
        x_conv_out is [word_batch_size, filters]
        where filters == e_word
        """
        x_conv = self.conv(x_reshaped)
        x_conv = F.relu(x_conv)
        x_conv_out = torch.max(x_conv, dim=2)[0]

        return x_conv_out

    ### END YOUR CODE

if __name__ == '__main__':
    word_batch_size = 10
    e_char = 7
    filters = 3
    m_word = 21

    c = CNN(e_char, filters)

    i = torch.rand(word_batch_size, e_char, m_word)
    o = c(i)
    assert o.shape == (word_batch_size, filters)

    # min word size (delimiter + char + delimiter)
    i = torch.rand(word_batch_size, e_char, 3)
    o = c(i)
    assert o.shape == (word_batch_size, filters)

    print('Tests passed!')
