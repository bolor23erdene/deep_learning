
from torchtext.datasets import IWSLT2017
train_iter, valid_iter, test_iter = IWSLT2017()
src_sentence, tgt_sentence = next(iter(train_iter))

# data preparation german language


# text cleaning preprocessing


# model setup
# input, embeddings, encoder, decoder, dense, output