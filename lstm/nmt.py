
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

# text cleaning preprocessing


# model setup
# input, embeddings, encoder, decoder, dense, output