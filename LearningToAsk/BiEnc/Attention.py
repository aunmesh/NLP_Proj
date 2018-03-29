from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from pytorch_misc import rnn_mask, packed_seq_iter, seq_lengths_from_pad,const_row


class Attention_Context(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
            '''
                Constructor for Attention_Context Class
                Input:
                class Attention_Context(nn.module):
                    encoder_dim     : Dimension of the hidden states of the encoder
                    decoder_dim     : Dimension of the hidden states of the decoder
            '''
            super(Attention_Context, self).__init__()

            self.encoder_dim = encoder_dim
            self.decoder_dim = decoder_dim

            # self.sim_matrix is the similarity matrix which is used for calculating attention energies
            self.sim_matrix = nn.Linear(self.decoder_dim, self.encoder_dim)

    def forward(self, decoder_states, encoder_states, mask=None):
        '''
        forward function:
            inputs:
                decoder_states      : Hidden state of the Decoder LSTM (Size: [ Batch_size , self.decoder_dim ] )
                encoder_states      : Hidden States of the Encoder Sequence ( Size: [Batch_Size, T, self.encoder_dim] )
                rnn_mask            : Mask is a boolean Matrix (Size : [Batch_Size , T ] ). Has 1 for each row where sequence elements are present and 0 where
                                      sequence elements are absent
            output:
                context_vec         : Attention weighted sum of the encoder hidden states ( Size: [Batch_Size , self.encoder_dim] )
                attention_energies  : Attention energies for each sequence ( Size: [ Batch_Size, T ] )

            Note: T represents length of the maximum length sequence in a Batch.
        '''
        B_size = decoder_states.size(0)

        assert (decoder_states.size(1) == self.decoder_dim), \
            "Input Decoder States feature length are not consistent with the values given to the Constructor!"
        assert(B_size == encoder_states.size(0)), \
            "encoder_states and decoder_states batch sizes are not consistent!"
        assert(self.encoder_dim == encoder_states.size(2)), \
            "Input Encoder States feature length are not consistent with the values given to the Constructor!"

        decoder_states = decoder_states.contiguos()
        # Transformed States of Decoder (Linear Transform) to bring it in the same dimension as encoder_dim
        decoder_transformed = self.sim_matrix(decoder_states)
        # Unsqueezing the 2D Matrix a 3D Matrix for Batch Multiplication Purposes
        decoder_transformed_mat = decoder_transformed.unsqueeze(2)

        # Attention Energies are calculated
        # Result is of Size [Batch_size , T]
        Attention_Energies = torch.bmm(encoder_states , decoder_transformed_mat).squeeze(2)

        if mask is not None:
            # Maximum Enerygy for each Sequence, Result is of Size [Batch_size*1]
            shift = Attention_Energies.max(1)[0]

            # Shifting Energies for Numerical Stability
            Energy_Shifted = Attention_Energies - shift.expand_as(Attention_Energies.transpose(0,1)).transpose(0,1)

            # Applying Mask after taking exponential so that unwanted indices are reduced to 0 energies,
            # transpose is being taken so that expand as operation can be performed
            Energies = Energy_Shifted.exp() * mask

            Energies_sum = Energies.sum(1).expand_as(Energies.transpose(0,1)).tranpose(0,1)
            alpha = torch.div(Energies, Energies_sum) #Size [batch_size, T]

        else:
            alpha = F.softmax(Attention_Energies) #Size [batch_size, T]

        Attended_Context = torch.bmm(alpha.unsqueeze(1), self.encoder_states).squeeze(1)

        return Attended_Context, alpha
