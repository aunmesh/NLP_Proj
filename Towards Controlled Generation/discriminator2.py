import torch
import torch.nn as nn
import torch.nn.functional as F
# from keras.preprocessing import sequence
# from keras.datasets import imdb

num_classes = 7

# Defining the Encoder
class Discriminator(nn.Module):
    ''' Discriminator is a neural network and hence a subclass of nn.Module.
        The aim of the discriminator is to classify attributes, based on sentences.
    '''
    def __init__(
                self,
                n_ques_vocab,
                maxlen=15,
                d_wordvec=50,
                dropout=0.1,
                usecuda=False
                ):
                super(Discriminator, self).__init__()

                # Useful links here:
                # Padding: https://github.com/nicolas-ivanov/tf_seq2seq_chatbot/issues/15
                # Using CNNs for text classification: http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
                self.src_word_emb = nn.Embedding(n_ques_vocab, d_wordvec, padding_idx=0)
                # Next few line copied verbatim from : https://github.com/GBLin5566/toward-controlled-generation-of-text-pytorch/blob/master/Model/Modules.py
                # This is the classification convnet
                # Dropout probability = dropout
                self.drop = nn.Dropout(p=dropout)
                # Length of input channel = maxlen (number of input maps, basically), number of maps created is 128 and each kernel looks at a 5 word window at a time.
                self.conv1 = nn.Conv1d(maxlen, 128, kernel_size=2)
                self.conv2 = nn.Conv1d(128, 128, kernel_size=2)
                self.conv3 = nn.Conv1d(128, 128, kernel_size=2)
                self.softmax = nn.LogSoftmax()


    def forward(self, input_sentence, is_softmax=False, dont_pass_emb=False):
        if dont_pass_emb:
            emb_sentence = input_sentence
        else:
            emb_sentence = self.src_word_emb(input_sentence)
        # print(emb_sentence.size())
        # print("dimension of embedding is ", emb_sentence)
        relu1 = F.relu(self.conv1(emb_sentence))
        layer1 = F.max_pool1d(relu1, 2)
        # print("shape of layer1 is ", layer1.size())
        relu2 = F.relu(self.conv2(layer1))
        layer2 = F.max_pool1d(relu2, 2)
        layer_temp = F.relu(self.conv2(layer2))
        layer3 = F.max_pool1d(layer_temp, 2)
        flatten = self.drop(layer2.view(layer3.size()[0], -1))
        if not hasattr(self, 'linear'):
            self.linear = nn.Linear(flatten.size()[1], num_classes)
            self.linear = self.linear
        logit = self.linear(flatten)
        if is_softmax:
            logit = self.softmax(logit)
        return logit
