
import time
from torch.nn.utils.rnn import pad_sequence
import torch
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

from model import LSTM

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
emsize = 128
hidden_dim = 64
output_dim = 64
num_classes = 4

# define device as a gpu or a cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a tokenizer
tokenizer = get_tokenizer('basic_english')


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


# define train_iter + create a vocab from train_iter
train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab(['here', 'is', 'an', 'example'])

num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)

print("num_class: ", num_class)
print("vocab_size: ", vocab_size)


def text_pipeline(x): return vocab(tokenizer(x))
def label_pipeline(x): return int(x) - 1


# collate lists of samples into batches
def collate_batch(batch):
    label_list, text_list, offsets = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets, dtype=torch.int64)
    #text_list = torch.tensor(text_list, dtype=torch.int64)
    # pad_sequence is the magical function
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# define the dataloader
dataloader = DataLoader(
    train_iter,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_batch)

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']

model = LSTM(
    vocab_size,
    emsize,
    hidden_dim,
    output_dim,
    n_layers=1,
    bidirectional=False,
    dropout_rate=0.2,
    pad_index=pad_index).to(device)

# model = LSTMTagger(
#     emsize,
#     hidden_dim,
#     vocab_size,
#     num_classes)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 10
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        if idx == 0:
            print('label: ', label.shape)
            print('text: ', text.shape)
            print('offset ', offsets.shape)

        optimizer.zero_grad()

        predicted_label = model(text, offsets)
        
        print("predicted_label", predicted_label.shape)
        print("predicted_label", label.shape)

        loss = criterion(predicted_label, label)

        loss.backward()
        optimizer.step()

        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc / total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.dataset import random_split

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = AG_NEWS()
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)

split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    print('-' * 59)
