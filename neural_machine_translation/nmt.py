"""SPACY: open source NLP library.
1. download the english and german language
2. create tokenizer 
"""
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive
import io

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

import spacy
spacy.load("en_core_web_sm")
spacy.load("de_core_news_sm")

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

print(spacy.__version__, torchtext.__version__, torch.__version__)

def build_vocab(filepath, tokenizer):
  counter = Counter()
  with io.open(filepath, encoding="utf8") as f:
    for string_ in f:
      counter.update(tokenizer(string_))
  return Vocab(counter)#, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

def data_process(filepaths):
  raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
  raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
  data = []
  for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
    de_tensor_ = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)],
                            dtype=torch.long)
    en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)],
                            dtype=torch.long)
    data.append((de_tensor_, en_tensor_))
  return data

train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)

"""Dataloaders"""
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128
PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
  de_batch, en_batch = [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
  de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
  en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
  return de_batch, en_batch

train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
test_iter = DataLoader(test_data, batch_size=BATCH_SIZE,
                       shuffle=True, collate_fn=generate_batch)

import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


#print(next(train_iter))
for _, (src, trg) in enumerate(train_iter):
    print(src.shape,trg.shape)


# class Encoder:
#     def __init__(self,):
#         self.Embedding = torch.Embedding
#         self.RNN = torch.GRU
        
        
#     def forward():

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = False)

        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim)

        
    def forward(self, src: Tensor) -> Tuple[Tensor]:

        # src = batch_size x max_seq_len
        embedded = self.embedding(src)

        # embedded = batch_size x max_seq_len x emb_dim
        outputs, hidden = self.rnn(embedded)

        # hidden[-1, :, :] = 1 x batch x enc_hid_dim 
        hidden = torch.tanh(self.fc(hidden[-1,:,:])) # last_sequence x batch x hid_dim

        # hidden = 1 x batch x dec_hid_dim
        return outputs, hidden
      
      
# training

encoder = Encoder(vocab_size=1000, embed_dim=32, enc_hid_dim=64, dec_hid_dim=64)
def train():
  for text in train_data_loader:
    # text = batch x max_seq_len
    outputs, hidden = encoder(text)
      
    # hidden = batch x dec_hid_dim
    decoder_ouputs  = decoder(hidden)  
    
    # decoder_outputs = batch x max_seq_len x |vocab|
    # labels = batch x max_seq_len 
    loss = criterion(decoder_outputs, labels)
      

    
class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 dropout: int,
                 attention: nn.Module):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU(enc_hid_dim, dec_hid_dim)

        self.out = nn.Linear(emb_dim, output_dim)

    def forward(self,
                input: Tensor,
                decoder_hidden: Tensor,
                encoder_outputs: Tensor) -> Tuple[Tensor]:

      embedded = self.embedding(input)
      
      rnn_output, hidden = self.rnn(embedded)

      # rnn_output = batch x max_seq_len x dec_hid_dim 
      outputs = self.out(rnn_output)
      
      # outputs = batch x max_seq_len x |vocab|
      return nn.softmax(outputs, axis=2)
    
      
      


# text cleaning preprocessing

# encoder = 


# decoder = 


# model setup
# input, embeddings, encoder, decoder, dense, output

### ENCODER 

### DECODER 
# for _, (src, trg) in enumerate(train_iter):
    
#     # src - input -> embedding layer -> lstm -> |vocab|
    
#     # input: batch x max_len x vocab_size
#     # embed: batch x max_len x emd_size
#     # lstm: 
#     # create predictions 
#     # create hidden
#     #predictions = torch.zeros(n, max_len, vocab_size)
#     for t in range(max_len):
        
#         lstm_input = batch x 1 x emb_size
#         out, (hidden, cell) = LSTM(lstm_input, (hidden, cell))
#         # hidden = batch x 1 x lstm_hidden_size
        
#         cur_preds = fc(hidden) # batch x 1 x lstm_hidden_size
        
#         predictions[prev_len:prev_len+batch_size, t, :] = cur_preds 
        
        
        