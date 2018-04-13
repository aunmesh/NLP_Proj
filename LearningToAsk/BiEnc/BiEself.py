from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from pytorch_misc import rnn_mask, packed_seq_iter, seq_lengths_from_pad, const_row

# Encoder Neural Net for encoding a Sequence using BiDirectional LSTM

##################################################
# Make this Batch First
###############################################


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, use_embedding, vocab_size, pad_idx=None, num_layers=2):
        """
            input_size: feature size of the input
            hidden_size: size of the hidden state of the LSTM
            use_embedding: true if we have to use embedding
            vocab_size: size of the vocabulary (needed only if use_embedding is true)
            pad_idx: pad_indices(needed only if use_embedding is true)
        """
        super(EncoderLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_embedding = use_embedding
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, bidirectional=True, batch_first = True)
	self.pad_idx = pad_idx

        if self.use_embedding :
            assert self.vocab_size is not None
            #assert self.pad_idx is not None
            self.pad_idx = pad_idx

            # Make sure this is Glove Embedding
            self.embed = nn.Embedding(self.vocab_size , self.input_size, padding_idx=pad_idx)

    def forward(self, x):
        """
            Input:
                x           :   input batch of sequences, Can be a PackedSequence
            Output:
                output_t    :   Concatenated BiDirectional Hidden States for each Variable in the Sequence
                                Shape: [ Batch_Size * Max_Seq_Length , 2 * self.hidden_size ]
                lengths     :   List Containing Lengths of the Sequences for each Sequence in the input batch x
                h_n_fixed   :   Final Hidden States for each Sequence in the Batch
                                Shape: [ Batch_Size , 2 self.hidden_size ]
        """


        if isinstance(x, PackedSequence):
            x_embed = x if self.use_embedding is False else PackedSequence(self.embed(x[0]), x[1])
        else:
            x_embed = x if self.use_embedding is False else self.embed(x)
        # output : [seq_len,batch,hidden_size * num_directions](if input not as PackedSeq)
        output, H_n = self.lstm(x_embed)

        # h_n, c_n are of Size [num_layers * num_directions, batch, hidden_size]
        h_n = H_n[0]  # Hidden States
        c_n = H_n[1]  # Cell States of the LSTM

        # h_n_fixed has the final hidden states for all the sequences in the batch
        # Size : [batch_size, num_layers * num_directions , 2 * hidden_size ]
        #h_n_fixed = h_n.transpose(0,1)#.contiguos()
        #c_n_fixed = c_n.transpose(0,1)#.contiguos()

        if isinstance(output, PackedSequence):
            output, output_lengths = pad_packed_sequence(output, batch_first=True)    # output : [max_seq_length, batch, 2 * hidden_size]

        else:
            output_lengths = [output.size(1)] * output.size(0)      # [length_of_sequence] * batch_size

        # output_t = output.transpose(0,1).view(-1, 2 * self.hidden_size) # Flattening the array

        # Dimension : [Batch_Size , Max_Seq_Length, 2 * hidden_size]
        #output_t = output.transpose(0,1)
	
	temp = max(output_lengths)
	mask = Variable(torch.zeros( output.size(0), temp), requires_grad=False).cuda()

	for ind,val in enumerate(output_lengths):
		mask[ind,:val] = 1

        return output, output_lengths, (h_n, c_n), mask
