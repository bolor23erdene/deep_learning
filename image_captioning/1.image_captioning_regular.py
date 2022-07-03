import torch.nn as nn
import torchvision.models as models
import torch

import os
import pandas
import spacy

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import transforms

spacy_eng = spacy.load("en")

class Vocabulary:
    def __init__(self, freq_threshold):
        
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        
        self.freq_threshold = freq_threshold
    
    def __len__(self):
        return len(self.itos)
    
    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]
    
    def build_vocabulary(self,sentences):
        idx = 4
        frequency = {}
        
        for sentence in sentences:
            for word in self.tokenizer_eng(sentence):
                if word not in frequency:
                    frequency[word] = 1
                else:
                    frequency[word] += 1
                
                if (frequency[word] > self.freq_threshold-1):
                    self.itos[idx] = word
                    self.stoi[word] = idx
                    idx += 1
    
    def numericalize(self,sentence):
        tokenized_text = self.tokenizer_eng(sentence)
        
        return [self.stoi[word] if word in self.stoi else self.stoi["<UNK>"] for word in tokenized_text ]
                    
        


class encoder(nn.Module):
    pass 
    
class decoder(nn.Module):
    pass 

class oneToMany(nn.Module):
    pass

# encoder CNN 
## input images: batch x 256 x 256 x 3 
## output embed: batch x 128 

# decoder RNN 
## input embed: batch x 128, batch x seq_len 
## output embed: 

# # image_embed_size = text_embed_size for convenience -> batch x hidden_size
# embedding, (hidden, cell) = self.lstm(image_embedding, (hidden, cell))

# pred = self.fc(hidden)

# predictions[1] = pred 

# embedding = pred

# for t in range(max_len):
    
#     # self.embed: batch x vocab_size -> batch x embed_size
#     text_embedding = self.embed(embedding)
    
#     # self.lstm: embed_size -> hidden_size
#     lstm_out, (hidden, cell) = self.lstm(embedding, hidden, cell)
    
#     # self.fc: batch x hidden_size -> batch x vocab_size
#     pred = self.fc(hidden)
    
#     # predictions: seq_len x batch x vocab_size
#     predictions[t] = pred 
    
#     # batch x 1 
#     embedding = pred
    