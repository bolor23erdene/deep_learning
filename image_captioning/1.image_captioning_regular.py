


# encoder CNN 
## input images: batch x 256 x 256 x 3 
## output embed: batch x 128 

# decoder RNN 
## input embed: batch x 128, batch x seq_len 
## output embed: 

# image_embed_size = text_embed_size for convenience -> batch x hidden_size
embedding, (hidden, cell) = self.lstm(image_embedding, (hidden, cell))

for t in range(max_len):
    
    # self.embed: batch x vocab_size -> batch x embed_size
    text_embedding = self.embed(embedding)
    
    # self.lstm: embed_size -> hidden_size
    lstm_out, (hidden, cell) = self.lstm(embedding, hidden, cell)
    
    # self.fc: batch x hidden_size -> batch x vocab_size
    pred = self.fc(hidden)
    
    # predictions: seq_len x batch x vocab_size
    predictions[t] = pred 
    
    # batch x 1 
    embedding = pred.argmax(1)
    
    