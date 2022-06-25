
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

from model import LSTM

embedding_dim = 16
hidden_dim = 8
n_layers = 1
bidirectional = False
dropout_rate = 0.2

# https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html#torch.nn.utils.rnn.pad_sequence
x = [torch.tensor([3,4,5,6]), torch.tensor([1,2]), torch.tensor([7])]
x_padded = pad_sequence(x, batch_first=True, padding_value=0)
text_lengths = torch.tensor([4, 2, 1])

print(x_padded)


# batch_size x seq_len
embedding = nn.Embedding(10, embedding_dim, 0)
embedded = embedding(x_padded)

print(embedded.shape)

# batch_size x seq_len x embed_dim
packed_embedded = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)    

print(packed_embedded)

lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                    dropout=dropout_rate, batch_first=True)

lstm_out, (hidden, cell) = lstm(packed_embedded) # sequence_len x 1 x 10=hidden_dim - there will be 5=seq_len hidden layers

print(hidden.shape)
