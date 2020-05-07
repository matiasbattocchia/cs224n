#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch.nn.functional as F

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    e_char = 50
    dropout_prob = .3

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.word_embed_size = word_embed_size

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab.char2id),
            embedding_dim=self.e_char,
            padding_idx=vocab.char_pad,
        )

        self.cnn = CNN(self.e_char, self.word_embed_size)
        self.highway = Highway(self.word_embed_size)

        ### END YOUR CODE

    def forward(self, x_padded):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        m_sent, sent_batch_size, m_word = x_padded.shape

        # x_padded [m_sent, sent_batch_size, m_word]
        x_padded = x_padded.reshape(m_sent * sent_batch_size, -1)
        # x_padded [word_batch_size, m_word]
        x_emb = self.embedding(x_padded)
        # x_emb [word_batch_size, m_word, e_char]
        x_reshaped = x_emb.transpose(1,2)
        # x_reshaped [word_batch_size, e_char, m_word]
        x_conv_out = self.cnn(x_reshaped)
        # x_conv_out [word_batch_size, e_word]
        x_highway = self.highway(x_conv_out)
        # x_highway [word_batch_size, e_word]
        x_word_emb = F.dropout(x_highway, p=self.dropout_prob)
        # x_word_emb [word_batch_size, e_word]
        x_word_emb = x_word_emb.reshape(m_sent, sent_batch_size, -1)
        # x_word_emb [m_sent, sent_batch_size, e_word]
        return x_word_emb

        ### END YOUR CODE

