from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from pytorch_misc import rnn_mask, packed_seq_iter, seq_lengths_from_pad, const_row
from Attention import Attention_Context


# vocab size, embedding_size, encoder_dim, decoder_hidden dimension, eos_token, bos_token


class DecoderMLP(nn.Module):
    def __init__(self, encoder_hidden_dim, mlp_hidden_dim, decoder_hidden_dim=600, target_vocab_size=28000):

        ############ NOTE ####################
        # 1. find mlp_hidden_dim. not given in paper
        # 2.
        # embedding, pad_idx, Wt_row_size, bos_token=0,
        #  eos_token=1, , decoder_dropout_prob=0.3, max_size=15, ):

        # Input Args:
        #     encoder_hidden_dim  : Hidden Dimension of The Encoder (To be used for attention while Decoding)
        #     mlp_hidden_dim : Hideen Dimension of the multi-layer perceptron acting on decoder output
        #     decoder_hidden_dim  : Hidden Dimension of The Decoder (To be used for attention while Decoding)
        #     target_vocab_size : target vocabulary size (different from the source in our case)

        # W_t : mlp_hidden_dim * (encoder_hidden_dim + decoder_hidden_dim)
        # W_s : target_embedding_dim * mlp_hidden_dim

        super(DecoderMLP, self).__init__()
        self.target_vocab_size = target_vocab_size

        self.enc_h_dim = encoder_hidden_dim

        self.dec_h_dim = decoder_hidden_dim

        self.mlp_h_dim = mlp_hidden_dim
        self.mlp_wt = nn.Linear(self.enc_h_dim + self.dec_h_dim, self.mlp_h_dim)
        self.mlp_ws = nn.Linear(self.mlp_h_dim, self.target_vocab_size)

    def forward(self, h_t, c_t):

        mlp_input = torch.cat((h_t, c_t), 1)
        y_temp0 = self.mlp_wt(mlp_input)
        y_temp1 = F.tanh(y_temp0)
        y_temp2 = F.softmax(y_temp1)
        y_new = self.mlp_ws(y_temp2)

        return y_new
