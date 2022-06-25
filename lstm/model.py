import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(1)

""" 
LSTM dimensions:

The first axis is the sequence itself.
The second indexes instances in the mini-batch.
The third indexes elements of the input. 
"""

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        print("embeds: ", embedded.shape)
        # 64, 127, 128
        
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)    
        
        lstm_out, (hidden, cell) = self.lstm(packed_embedded) # sequence_len x 1 x 10=hidden_dim - there will be 5=seq_len hidden layers
        print("lstm_out: ", lstm_out.shape)
        # 64, 127, 64 
        # hidden -> 64, 1, 64
        #lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        #output, output_lengths = nn.utils.rnn.pad_packed_sequence(lstm_out)
        
        hidden = self.dropout(torch.cat(hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        pred = self.fc(hidden)
        print("pred: ", pred.shape)
    #     # 64, 127, 4
    #     pred = pred.view(64, -1)
        
    #    # pred = pred[:, -1]
        
        return pred
    
class LSTM_ATTN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional,
                 dropout_rate, pad_index):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout_rate, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    """
    https://linuxpip.org/pytorch-squeeze-unsqueeze/
    Simply put, torch.unsqueeze "adds" a superficial 1 dimension to tensor (at the specified dimension), 
    while torch.squeeze removes all superficial 1 dimensions from tensor.
    Below is a visual representation of what squeeze/unsqueeze do for an 2D matrix.
    """
    def attention(self, lstm_output, final_state):
        #print("lstm_output: ", lstm_output.shape)
        #print("final_state: ", final_state.shape)
        #lstm_output = lstm_output.permute(1, 0, 2) # (batch, seq_len, hid)-> (seq_len, batch, hid)
        
        # final_state= (1,batch,hidden)
        #merged_state = torch.cat([s for s in final_state], 1) # (seq_len, hid)
        
        # squeeze=Returns a tensor with all the dimensions of input of size 1 removed.
        # merged_state.squeeze(0) -> (batch,hidden)
        # merged_state.squeeze(0).unsqueeze(2) -> (batch,hidden,1)
        #merged_state = merged_state.squeeze(0).unsqueeze(2) 
        final_state = final_state.permute(1, 2, 0)
        #print("modified final_state: ", final_state.shape)
        #batch x matrix x matrix
        weights = torch.bmm(lstm_output, final_state) # (batch, seq_len, hidden) x (batch, hidden, 1) = (batch, seq, 1)
        #print("weights bmm(lstm_out, hidden): ", weights.shape)   
        weights = F.softmax(weights.squeeze(2), dim=1).unsqueeze(2) # batch x seq_len x 1
        #print("F.softmax(weights.squeeze(2), dim=1).unsqueeze(2): ", weights.shape) 
        att_output = torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2) # b x h x s x (b x s x 1) = b x h 
        #print("torch.bmm(torch.transpose(lstm_output, 1, 2), weights).squeeze(2): ", att_output.shape)   
        return att_output
    
    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths=length.cpu(), batch_first=True, 
                                                           # enforce_sorted=False)
        
        output, (hidden, cell) = self.lstm(embedded) ## n_layer x batch_size x hidden_dim
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        #output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output) ## output = [batch size, seq len, hidden dim * n directions]
        
        # x = self.fc1(output) #batch, seq_len, hid_dim
        # x = self.tanh(x)
        # x = self.soft(x)
        #context = torch.matmul(output, alpha)
        
        #context = torch.matmul(output, x) # batch, seq_len, hid_dim - batch, seq_len, 1
        # if self.lstm.bidirectional:
        #     hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
        #     # hidden = [batch size, hidden dim * 2]
        # else:
        #     hidden = self.dropout(hidden[-1])
            #hidden = [batch size, hidden dim]
        #prediction = self.fc(hidden)
        #prediction = self.fc(context)
        # prediction = [batch size, output dim]
        
        att_output = self.attention(output, hidden)
        pred = self.fc(att_output.squeeze(0))
        print(att_output.shape, att_output.squeeze(0).shape)
        
        
        return pred#prediction
    
    
    

class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_classes):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, num_classes)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1)) # sequence_len x 1 x 10=hidden_dim - there will be 5=seq_len hidden layers
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) #in: 5x10 hidden: 10x3 out: 5x3
        tag_scores = F.log_softmax(tag_space, dim=1) #in 5x3 #out 5x3
        
        print("embeds: ", embeds.shape)
        print("lstm_out: ", lstm_out.shape)
        print("tag_space: ", tag_space.shape)
        print("tag_scores: ", tag_scores.shape)
        
        return tag_scores
