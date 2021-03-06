import torchtext
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
from torchtext.utils import download_from_url, extract_archive
import io
import spacy

#!python --version
torchtext.__version__, torch.__version__, spacy.__version__

### download the training dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
train_urls = ('train.de.gz', 'train.en.gz')
val_urls = ('val.de.gz', 'val.en.gz')
test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

#!python -m spacy download de_core_news_sm

import spacy
spacy.load("en_core_web_sm")
spacy.load("de_core_news_sm")

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

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

de_vocab.get_itos()[0],de_vocab.get_itos()[1],de_vocab.get_itos()[2],de_vocab.get_itos()[3],de_vocab.get_itos()[4],de_vocab.get_itos()[-1]

en_vocab.get_itos()[0],en_vocab.get_itos()[1],en_vocab.get_itos()[2],en_vocab.get_itos()[3],en_vocab.get_itos()[4],en_vocab.get_itos()[-1]

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

len(train_data[0]), len(val_data), len(test_data)

for i in range(5):
    de, eng = train_data[i]
    print(eng.shape, de.shape)

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

class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, eng_vocab_size, n_layers, bidirectional, pad_idx):
        super().__init__()
        self.embedder = nn.Embedding(eng_vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers, bidirectional=bidirectional, batch_first=True)
        
    def forward(self, input, input_lens):
        # input = [seq_len x batch x eng_vocab_size]
        # print(input.shape, len(input_lens))
        embedded = self.embedder(input)#.permute(1, 0, 2)
        # print('embedded: ', embedded.shape)
        
        packed_embedded = pack_padded_sequence(embedded, input_lens.cpu(), batch_first=True, enforce_sorted=False)  
        
        # packed_embedded = [seq_len x batch x emb_dim]
        output, (hidden, cell) = self.rnn(packed_embedded)
        
        # output = [seq_len x batch x enc_hid_dim]
        # hidden = [1 x batch x enc_hid_dim]
        # cell =   [1 x batch x enc_hid_dim]
        
        return hidden, cell

# implement teacher force !!! 

class Decoder(nn.Module):
    def __init__(self, de_vocab_dim, dec_hid_dim, emb_dim, n_layers, bidirectional):
        super().__init__()
        self.de_vocab_dim = de_vocab_dim
        self.embedding = nn.Embedding(de_vocab_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, dec_hid_dim, n_layers, bidirectional=bidirectional, batch_first=False)
        self.fc = nn.Linear(dec_hid_dim, de_vocab_dim)
    
    def forward(self, input, hidden, cell):
        
        # input = 1 x batch 
        #input = input.unsqueeze(0)
        
        # 1 input = [128]
        # 2 input = [128, 19215] 
        # print("decoder input: ", input.shape)
        
        
        # embedded = 1 x batch x emb_dim 
        embedded = self.embedding(input).unsqueeze(0)
        
        # # 1 embedded = [1, 32, 128]
        # print("decoder embedded: ", embedded.shape)
        
        # # 1 decoder input hidden = []
        # print("decoder input hidden: ", hidden.shape)
        
        # hidden = 1 x batch x dec_hid_dim & (expects: 1 x 29 x 32) 
        out, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # 1 decoder input hidden = []
        # print("decoder input hidden: ", hidden.shape)
        
        
        # hidden = 1 x batch x dec_hidden_dim or (out = 1 x batch x hid_dim). In fact: out == hidden 
        output = self.fc(hidden)
    
        # 1 output = 1 x 128 x 19215
        # print("decoder output: ", output.shape)
        
        # output = 1 x batch x de_vocab_dim
        #output = nn.softmax(output, 2)
        
        return output, (hidden, cell)

# implement teacher force !!! 

from numpy.random import choice
elements = [1,0]
weights = [1,0]
teacher_force = 0

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder 
        self.decoder = decoder 
        self.device = device
        
    def forward(self, en_batch, en_text_lens, de_batch):
        # input_in_eng = seq_len x batch_size 
        seq_len = de_batch.size(1)
        batch_size = de_batch.size(0)
        de_vocab_dim = self.decoder.de_vocab_dim
        
        hidden, cell = self.encoder(en_batch, en_text_lens)

        outputs = torch.zeros(seq_len, batch_size, de_vocab_dim).to(self.device)# seq_len x batch x de_vocab_dim
    
        # de_batch = 
        input_decoder = de_batch[:, 0]
    
        for t in range(1, seq_len):
            
            output, (hidden, cell) = self.decoder(input_decoder, hidden, cell)

            # print("output: ", output.shape)
            
            #output = output.permute(1, 0, 2)

            # output = batch x 1 x |vocab| after permute & get rid of the dim=1
            outputs[t] = output.squeeze(0)
            
            # print("t: ", t, outputs.shape)
            # print(output.shape)
            # print(output.argmax(0).shape, output.argmax(1).shape, output.argmax(2).shape)
            
            # output = 1 x batch x vocab


            #print(choice(elements, p=weights))

            if teacher_force == choice(elements, p=weights):
                input_decoder = de_batch[:, t]
            else:
                input_decoder = output.argmax(2).squeeze(0)
            
            # predicted class or token from the predictions

            #input_decoder = output.argmax(2).squeeze(1)
            
        return outputs.permute(1,0,2)

import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

#from models import Encoder, Decoder, Seq2Seq
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
#start_time = time.time()

for idx, (en_batch, de_batch, en_text_lens, de_text_lens) in enumerate(train_iter):
    
    
    optimizer.zero_grad()
        
    predictions = seq2seq(en_batch, en_text_lens, de_batch)
    
    #print("loss: ", predictions.shape, de_batch.shape)
    
    # predictions = predictions.view(-1, de_vocab_size)
    # de_batch = de_batch.view(-1)
    
    predictions = torch.reshape(predictions, (-1, de_vocab_size))
    de_batch = torch.reshape(de_batch, (-1,))
    
    
    
    #print("loss: ", predictions.shape, de_batch.shape)
    
    loss = criterion(predictions, de_batch)
    # loss = criterion(predictions.argmax(2).squeeze(2).permute(1,0), de_batch)


    # loss = criterion(hidden, label)

    loss.backward()
    optimizer.step()

    if idx % 10 == 0:

        total_acc += (predictions.argmax(1) == de_batch).sum().item()
        total_count += de_batch.size(0)
        
        print(loss)
        print(int(100 * total_acc/total_count))

        print("Saving a model: ")
        torch.save(encoder.state_dict(), 'encoder.pth')
        torch.save(decoder.state_dict(), 'decoder.pth')
        print(choice(elements, p=weights))

    if idx == 20:
        break



"""# evaluation"""

def create_sentence(sentence, language):
    if language == 'en':
        get_word = en_vocab.get_itos()
    else:
        get_word = de_vocab.get_itos()

    ans = []
    for word in sentence:
        ans.append(get_word[word])

    return ans

import pandas as pd
def evaluation():
    with torch.no_grad():

        total_acc = 0
        total_count = 0
        step = 0

        encoder = Encoder(emb_dim=64, enc_hid_dim=32, eng_vocab_size=eng_vocab_size, n_layers=1, bidirectional=False, pad_idx=EN_PAD_IDX).to(device)
        decoder = Decoder(de_vocab_dim=de_vocab_size, dec_hid_dim=32, emb_dim=32, n_layers=1, bidirectional=False).to(device)
        encoder.load_state_dict(torch.load('encoder.pth'))
        decoder.load_state_dict(torch.load('decoder.pth'))
        encoder.eval()
        decoder.eval()

        steps = []
        losses = []
        acc = []

        for idx, (en_batch, de_batch, en_text_lens, de_text_lens) in enumerate(test_iter):

            seq2seq = Seq2Seq(encoder, decoder, device)

            predictions = seq2seq(en_batch, en_text_lens, de_batch)
            
            #print("loss: ", predictions.shape, de_batch.shape)
            
            # predictions = predictions.view(-1, de_vocab_size)
            # de_batch = de_batch.view(-1)
            
            predictions = torch.reshape(predictions, (-1, de_vocab_size))
            de_batch = torch.reshape(de_batch, (-1,))
            
            
            
            #print("loss: ", predictions.shape, de_batch.shape)
            
            loss = criterion(predictions, de_batch)
            # loss = criterion(predictions.argmax(2).squeeze(2).permute(1,0), de_batch)

            translating_preds = predictions.argmax(1)
            eng_sen = en_batch[0]
            de_sen = translating_preds[:de_text_lens[0]]

            #print("eng_sen: ", eng_sen)
            #print("de_sen: ", de_sen)

            en_sentence = create_sentence(eng_sen, 'en')
            de_sentence = create_sentence(de_sen, 'de')

            print("ENGLISH: ", ' '.join(en_sentence))
            print("GERMANY: ", ' '.join(de_sentence))
            
    
            # loss = criterion(hidden, label)

            # loss.backward()
            # optimizer.step()

            

            if idx % 1 == 0:
                print(idx)

                total_acc += (predictions.argmax(1) == de_batch).sum().item()
                total_count += de_batch.size(0)
            
                print("LOSS: ", loss)
                print("ACC: ", total_acc/total_count)


                steps.append(step)
                losses.append(loss.cpu().detach().numpy())
                acc.append(int(100*total_acc/total_count))
                step += 1

        df_results = pd.DataFrame({'step': step, 'acc': acc, 'loss': losses})
        df_results.to_csv('simplest_results.csv')


evaluation()
