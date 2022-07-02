"""SPACY: open source NLP library.
1. download the english and german language
2. create tokenizer 
"""
import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

de_vocab = build_vocab(train_filepaths[0], de_tokenizer)
en_vocab = build_vocab(train_filepaths[1], en_tokenizer)

de_vocab.set_default_index(de_vocab["<unk>"])
en_vocab.set_default_index(en_vocab["<unk>"])

print(en_vocab)
print(de_vocab)
print('en', len(en_vocab), len(de_vocab))


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



BATCH_SIZE = 128
DE_PAD_IDX = de_vocab['<pad>']
BOS_IDX = de_vocab['<bos>']
EOS_IDX = de_vocab['<eos>']
EN_PAD_IDX = en_vocab['<pad>']

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def generate_batch(data_batch):
  de_batch, en_batch, en_text_lens, de_text_lens = [], [], [], []
  for (de_item, en_item) in data_batch:
    de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    en_text_lens.append(en_item.size(0))
    de_text_lens.append(de_item.size(0))
  de_batch = pad_sequence(de_batch, batch_first = True, padding_value=DE_PAD_IDX)
  en_batch = pad_sequence(en_batch, batch_first = True, padding_value=EN_PAD_IDX)   
  en_text_lens = torch.tensor(en_text_lens, dtype=torch.int64)
  de_text_lens = torch.tensor(de_text_lens, dtype=torch.int64)
  
  return en_batch.to(device), de_batch.to(device), en_text_lens.to(device), de_text_lens.to(device)

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

from models import Encoder, Decoder, Seq2Seq
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split
import time

eng_vocab_size = len(en_vocab)
de_vocab_size = len(de_vocab)

encoder = Encoder(emb_dim=64, enc_hid_dim=32, eng_vocab_size=eng_vocab_size, n_layers=1, bidirectional=False, pad_idx=EN_PAD_IDX).to(device)
decoder = Decoder(de_vocab_dim=de_vocab_size, dec_hid_dim=32, emb_dim=32, n_layers=1, bidirectional=False).to(device)
seq2seq = Seq2Seq(encoder, decoder, device).to(device)

LR = 5

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(encoder.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
# total_accu = None

# encoder.train()
# decoder.train()
total_acc, total_count = 0, 0
log_interval = 10
start_time = time.time()

for idx, (en_batch, de_batch, en_text_lens, de_text_lens) in enumerate(train_iter):
    
    
    optimizer.zero_grad()
        
    predictions = seq2seq(en_batch, en_text_lens, de_batch)
    
    print("loss: ", predictions.shape, de_batch.shape)
    
    # predictions = predictions.view(-1, de_vocab_size)
    # de_batch = de_batch.view(-1)
    
    predictions = torch.reshape(predictions, (-1, de_vocab_size))
    de_batch = torch.reshape(de_batch, (-1,))
    
    
    
    print("loss: ", predictions.shape, de_batch.shape)
    
    loss = criterion(predictions, de_batch)
    # loss = criterion(predictions.argmax(2).squeeze(2).permute(1,0), de_batch)


    # loss = criterion(hidden, label)

    loss.backward()
    optimizer.step()

    total_acc += (predictions.argmax(1) == de_batch).sum().item()
    total_count += de_batch.size(0)
    
    print(loss)
    print(total_acc/total_count)

        # elapsed = time.time() - start_time
        # print('| epoch {:3d} | {:5d}/{:5d} batches '
        #         '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
        #                                     total_acc / total_count))
        # total_acc, total_count = 0, 0
        # start_time = time.time()