
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import pandas as pd
import spacy
import random
from torchtext.data.metrics import bleu_score
from pprint import pprint
from torch.utils.tensorboard import SummaryWriter

# data preparation german language
spacy_german = spacy.load("de")
spacy_english = spacy.load("en")

def tokenize_german(text):
    return [token.text for token in spacy_german.tokenizer(text)]

def tokenize_english(text):
    return [token.text for token in spacy_english.tokenizer(text)]

### Sample Run ###

sample_text = "I love machine learning"
print(tokenize_english(sample_text))

german = Field(tokenize=tokenize_german,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")

english = Field(tokenize=tokenize_english,
               lower=True,
               init_token="<sos>",
               eos_token="<eos>")

train_data, valid_data, test_data = Multi30k.splits(exts = (".de", ".en"),
                                                    fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=3)
english.build_vocab(train_data, max_size=10000, min_freq=3)

print(f"Unique tokens in source (de) vocabulary: {len(german.vocab)}")
print(f"Unique tokens in target (en) vocabulary: {len(english.vocab)}")

# dir(english.vocab)
print(english.vocab.__dict__.keys())
print(list(english.vocab.__dict__.values()))
e = list(english.vocab.__dict__.values())
for i in e:
  print(i)
  
word_2_idx = dict(e[3])
idx_2_word = {}
for k,v in word_2_idx.items():
  idx_2_word[v] = k

# text cleaning preprocessing


# model setup
# input, embeddings, encoder, decoder, dense, output