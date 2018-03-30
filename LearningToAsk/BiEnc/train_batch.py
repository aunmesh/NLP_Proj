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
from torch import optim
from Attention import Attention_Context
from BiEself import EncoderLSTM
from DecoderNew import DecoderLSTM
from DecoderMLP import DecoderMLP

def get_token(softmax_vector):

    return None

def train_batch(batch, encoder, decoder, attention, mlp, criterion, optimizers, max_output_size):

    for opt in optimizers:
        opt.zero_grad()

    # DEFINE THESE
    eh_0 = None             # [batch_size, encoder_hidden_dim]
    ec_0 = None             # [batch_size, encoder_hidden_dim]
    encoder_initial_state = (eh_0, ec_0)

    # encoder_output : [Batch_Size , Max_Seq_Length, 2 * hidden_size]

    encoder_output, encoder_output_len, encoder_states = encoder(batch, encoder_initial_state)

    encoder_hidden = encoder_states[0]              # extract encoder hidden states
    dh_0 = torch.cat((encoder_hidden[0], encoder_hidden[1]), 1)
    dc_0 = None         # DEFINE THIS
    decoder_initial_state = (dh_0, dc_0)
    first_token = None # DEFINE THIS: starting token for the question <SOS>token embedding

    for step in range(max_output_size):
        if step == 0:
            decoder_output, decoder_states = decoder(first_token, decoder_initial_state)
        else:
            decoder_output, decoder_states = decoder(prev_token, decoder_prev_state)

        context_attention, __ = attention(encoder_output, decoder_output)

        softmax_vector = mlp(decoder_output, context_attention)
        next_token = get_token(softmax_vector)

        prev_token = next_token
        decoder_prev_state = decoder_states




        # Train_algo:
        # 1. run encoder
        # 2. get h_0 for decoder from encoder output
        # 3. decoder lstm one step
        # 4. attention context one step (encoder hidden state, decoder hidden state)
        # 5. run mlp one step
        # 6. goto step 3 till Max_Output_Size
