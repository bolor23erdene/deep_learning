import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


import numpy as np
import pandas as pd

import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

train_path = "train.csv"
test_path = "test.csv"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df = train_df.drop(columns=["Id"])
train_df = train_df.dropna()
train_df = train_df[train_df['Tweet'] != "Not Available"]
train_df

test_df = test_df.rename(columns={"Category": "Tweet"})

test_df = test_df.drop(columns=["Id"])
test_df = test_df.dropna()
test_df = test_df[test_df['Tweet'] != "Not Available"]
test_df

train_df['Category'].value_counts()