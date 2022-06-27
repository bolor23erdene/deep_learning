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
        print(input.shape, len(input_lens))
        embedded = self.embedder(input)#.permute(1, 0, 2)
        print('embedded: ', embedded.shape)
        
        packed_embedded = pack_padded_sequence(embedded, input_lens.cpu(), batch_first=True, enforce_sorted=False)  
        
        # packed_embedded = [seq_len x batch x emb_dim]
        output, (hidden, cell) = self.rnn(packed_embedded)
        
        # output = [seq_len x batch x enc_hid_dim]
        # hidden = [1 x batch x enc_hid_dim]
        # cell =   [1 x batch x enc_hid_dim]
        
        return hidden, cell
        
        
    
    
    
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
        
        print("decoder input: ", input.shape)
        
        # embedded = 1 x batch x emb_dim 
        embedded = self.embedding(input).unsqueeze(0)
        
        print("decoder embedded: ", embedded.shape)
        
        print("decoder input hidden: ", hidden.shape)
        
        # hidden = 1 x batch x dec_hid_dim & (expects: 1 x 29 x 32) 
        out, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        print("decoder input hidden: ", hidden.shape)
        
        print(hidden)
        print(out)
        
        # hidden = 1 x batch x dec_hidden_dim or (out = 1 x batch x hid_dim). In fact: out == hidden 
        output = self.fc(hidden)
    
        
        print("decoder output: ", output.shape)
        
        # output = 1 x batch x de_vocab_dim
        #output = nn.softmax(output, 2)
        
        return output, (hidden, cell)
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder 
        self.decoder = decoder 
        self.device = device
        
    def forward(self, en_batch, en_text_lens, de_batch):
        # input_in_eng = seq_len x batch_size 
        seq_len = en_batch.size(0)
        batch_size = en_batch.size(1)
        de_vocab_dim = self.decoder.de_vocab_dim
        
        hidden, cell = self.encoder(en_batch, en_text_lens)

        outputs = torch.zeros((seq_len, batch_size, de_vocab_dim)).to(self.device)# seq_len x batch x de_vocab_dim
    
        # de_batch = 
        input_decoder = de_batch[:, 0]
    
        for t in range(1, seq_len):
            
            output, (hidden, cell) = self.decoder(input_decoder, hidden, cell)

            outputs[t, :, :] = output
            
            # predicted class or token from the predictions
            input_decoder = output.argmax(1)
            
        return outputs
            
            
            
# encoder = Encoder()
# decoder = Decoder()
# seq2seq = Seq2Seq(encoder, decoder, de_vocab_size, de_output)
            
# def train(dataloader):
    
    
#     for idx, (label, text, text_lengths) in enumerate(dataloader):
        
#         outputs = seq2seq(text)
        
#         loss = criterion(outputs, label)
        
#         loss.backward()
        
#         optimizer.step()
        
#         total_acc += 
#         total_count += label.size(0)
        
#         print("acc: ", total_acc/total_count)
        
        
        
            
    