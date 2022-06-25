
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

import unittest
import os

embedding_dim = 16
hidden_dim = 8
n_layers = 1
bidirectional = False
dropout_rate = 0.2
vocab_size = 10

class TestDataPreparation(unittest.TestCase):

    def setUp(self):
        self.base_fixture = os.path.dirname(__file__)

    # test each function
    def test_padding(self):
        """Validate if the padding is working as expected."""
        print(self.test_padding.__doc__)
        
        # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html#torch.nn.utils.rnn.pad_sequence
        # each index of the torch tensor represents a word in a sentence
        # for a given example, there are three sentences and seven words in total
        
        x = [torch.tensor([3,4,5,6]), torch.tensor([1,2]), torch.tensor([2])]
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        text_lengths = torch.tensor([4, 2, 1]) # keeping track of the text lengths will be useful for training to not do prediction for padded values. 
        print(x_padded)

        self.assertEqual(x_padded.shape[1], 4,
                         "after padding, the tensor shape dim 1 value should be equal to the max_len of the longest sentence")


    def test_embedding(self):
        """Validate if embedding is working as expected."""
        print(self.test_embedding.__doc__)
        
        x = [torch.tensor([3,4,5,6]), torch.tensor([1,2]), torch.tensor([2])]
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        text_lengths = torch.tensor([4, 2, 1]) # keeping track of the sentence lengths
        
        # batch_size x seq_len # padded in the previous step
        embedding = nn.Embedding(vocab_size, embedding_dim, 0)
        embedded = embedding(x_padded)
        # 3 x 4 x 16
        print(embedded)

        self.assertEqual(embedded.shape[2], embedding_dim,
                         "embedding layer dimension should be embedding_dim")

    def test_pack_padded_seq(self):
        """Validate if pack_padded_sequence is working properly."""
        print(self.test_pack_padded_seq.__doc__)
        
        x = [torch.tensor([3,4,5,6]), torch.tensor([1,2]), torch.tensor([2])]
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        text_lengths = torch.tensor([4, 2, 1]) # keeping track of the sentence lengths
        
        # batch_size x seq_len # padded in the previous step
        embedding = nn.Embedding(vocab_size, embedding_dim, 0)
        embedded = embedding(x_padded)

        print(embedded)
        
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)    

        print(packed_embedded.data.shape)

        self.assertEqual(packed_embedded.data.shape[0], 7,
                         "total number of words in all input sentences")
        
        
    def test_lstm(self):
        """Validate the dimensions of the LSTM outputs. """
        print(self.test_lstm.__doc__)
        
        x = [torch.tensor([3,4,5,6]), torch.tensor([1,2]), torch.tensor([2])]
        x_padded = pad_sequence(x, batch_first=True, padding_value=0)
        text_lengths = torch.tensor([4, 2, 1]) # keeping track of the sentence lengths
        
        # batch_size x seq_len # padded in the previous step
        embedding = nn.Embedding(vocab_size, embedding_dim, 0)
        embedded = embedding(x_padded)

        print(embedded)
        
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)    

        print(packed_embedded.data.shape)
        
        lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)

        lstm_out, (hidden, cell) = lstm(packed_embedded) # sequence_len x 1 x 10=hidden_dim - there will be 5=seq_len hidden layers

        #print('lstm: ', lstm_out.shape) # max_seq_len x batch x hidden_dim
        print('hidden: ', hidden.shape) # 1 x batch x hidden_dim
        
        print('hidden[-1, :, :]: ', hidden[-1, :, :].shape)

        self.assertEqual(hidden.data.shape[0], 1,
                         "last word of each sequence")
        
        self.assertEqual(hidden.data.shape[1], 3,
                         "batch size")
        
        self.assertEqual(hidden.data.shape[2], 8,
                         "hidden dimension size")





# x = [torch.tensor([3,4,5,6]), torch.tensor([1,2]), torch.tensor([7])]
# x_padded = pad_sequence(x, batch_first=True, padding_value=0)
# text_lengths = torch.tensor([4, 2, 1])

# print(x_padded)


# # batch_size x seq_len
# embedding = nn.Embedding(10, embedding_dim, 0)
# embedded = embedding(x_padded)

# print(embedded.shape)

# # batch_size x seq_len x embed_dim
# packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)    

# print(packed_embedded)
# print(len(packed_embedded))
# print(packed_embedded.data.shape)

# lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
#                     dropout=dropout_rate, batch_first=True)

# lstm_out, (hidden, cell) = lstm(packed_embedded) # sequence_len x 1 x 10=hidden_dim - there will be 5=seq_len hidden layers

# print(hidden.shape)
