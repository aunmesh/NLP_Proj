import torch
import torch.nn as nn


# Defining the Encoder
class Discriminator(nn.Module):
    ''' Discriminator is a neural network and hence a subclass of nn.Module.
        The aim of the discriminator is to classify attributes, based on sentences.
    '''
    def __init__(
                self,
                n_ques_vocab,
                maxlen,
                dropout=
                d_wordvec,
                usecuda=False
                ):
                super(Discriminator, self).__init__()

                # Useful links here:
                # Padding: https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15
                # Using CNNs for text classification: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
                self.word_embedding = nn.Embedding(n_ques_vocab, d_wordvec, padding_idx=0)
                # Next few line copied verbatim from : https://github.com/GBLin5566/toward-controlled-generation-of-text-pytorch/blob/master/Model/Modules.py
                # This is the classification convnet
                # Dropout probability = dropout
                self.drop = nn.Dropout(p=dropout)
                # Length of input channel = maxlen (number of input maps, basically), number of maps created is 128 and each kernel looks at a 5 word window at a time.
                self.conv1 = nn.Conv1d(maxlen, 128, kernel_size=5)
                self.conv2 = nn.Conv1d(128, 128, kernel_size=5)
                self.conv3 = nn.Conv1d(128, 128, kernel_size=5)
                self.softmax = nn.LogSoftmax()
