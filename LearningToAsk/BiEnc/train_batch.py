from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
import numpy as np

target_vocab_size = 28442
input_size = 300
target_embed = nn.Embedding(target_vocab_size , input_size, padding_idx=-1)  # Make sure this is Glove Embeddings

tgt_embedding_file = '../Data/source_target_output.npy'
tgt_embedding = torch.from_numpy(np.load(tgt_embedding_file))

target_embed.weight = nn.Parameter(tgt_embedding )

def get_next_token_indices(softmax_vector):
    required_indices = torch.max(softmax_vector,1)[1]
    return required_indices

def get_embedding(indices):
    return target_embed( indices )


def get_ground_truth_probab(softmax_vector, question_indices):

    return softmax_vector.gather(1, question_indices.unsqueeze(1) )

'''
Args:
    batch : batch is a tuple carrying Glove indices of the source and question for each sequence
'''


def train_batch(batch, encoder, decoder, attention, mlp,  optimizers, max_output_size):

    for opt in optimizers:
        opt.zero_grad()

    # Encoder Initialisations
    eh_0 = None                                     # [batch_size, num_layer * num_directions , encoder_hidden_dim]
    ec_0 = None                                     # [batch_size, num_layer * num_directions , encoder_hidden_dim]
    encoder_initial_state = (eh_0, ec_0)

    source_batch_indices = batch[0]                 # a packed Sequence as Source passages can be of different lengths
    
    # question_batch_indices = batch[1]               # a packed Sequence as Target passages can be of different lengths
    # question_indices_batch, question_lengths = pad_packed_sequence(question_batch_indices, batch_first=True, padding_value=0) # output is of size [Max_seq_length , batch , 2 * hidden_size]

    question_indices_batch = batch[1][0]
    question_lengths = batch[1][1]

    batch_size = len(question_lengths)

    # Ignoring initial Encoder states for now, encoder_output : [Batch_Size , Max_Seq_Length, 2 * hidden_size]

    encoder_output, encoder_output_len, encoder_states, mask_source = encoder(source_batch_indices)
    encoder_hidden_states = encoder_states[0]       # [batch, num_layers * num_directions, hidden_size]
    
    encoder_cell_states = encoder_states[1]

    dh_0_1 = torch.cat((encoder_hidden_states[0], encoder_hidden_states[1]), 1)
    dh_0_2 = torch.cat((encoder_hidden_states[2], encoder_hidden_states[3]), 1)
    dh_0 = torch.stack((dh_0_1 , dh_0_2), 0)

    dc_0_1 = torch.cat((encoder_cell_states[0], encoder_cell_states[1]), 1)
    dc_0_2 = torch.cat((encoder_cell_states[2], encoder_cell_states[3]), 1)
    dc_0 = torch.stack((dc_0_1 , dc_0_2), 0)

    
    # dc_0 = torch.zeros( dh_0.size() )
    
    decoder_prev_state = (dh_0, dc_0)
    #prev_token = None                              # DEFINE: <SOS>token embedding
    prev_token = Variable(-1* torch.ones(batch_size,1,300))	# Definition matches with <SOS> embedding defined in data

    prev_token = get_embedding(question_indices_batch[:, 0]).unsqueeze(1)

    # Final Size : [Batch_Size, Max_Seq_Length]
    total__probab__tensor = Variable(torch.ones((batch_size, 1)))
    dummy = 0
    for step in range(1, max_output_size):

        decoder_next_output, decoder_next_state = decoder(prev_token, decoder_prev_state)
        context_vec , __ = attention(encoder_output, decoder_next_output, mask_source)

        # softmax_vector : [ Batch_Size , Target_Vocab_Size ]
        softmax_vector = mlp(decoder_next_output, context_vec)
        next_token = get_embedding( get_next_token_indices(softmax_vector) )

        # In the Loss Calculation We only need the probability of the actual Ground truth. Note

	# If the step increases beyond the maximum length of any question in this batch, then we do slicing using a dummy variable. This won't matter as there is no contribution of these in final loss
	try:
		temp = get_ground_truth_probab(softmax_vector, question_indices_batch[:, step])
	except:
		temp = get_ground_truth_probab(softmax_vector, question_indices_batch[:, dummy])

        total__probab__tensor = torch.cat( (total__probab__tensor, temp), 1)

        prev_token = next_token.unsqueeze(1)                     # List of next embeddings for the decoder
        decoder_prev_state = decoder_next_state


    loss = 0

    for o,l in zip( total__probab__tensor, question_lengths ):
        temp = -1*torch.log(o[:l])
        
        normalizing_factor = l * batch_size
        temp_sum = torch.sum(temp) / normalizing_factor

        loss+=temp_sum

    loss.backward()

    for o in optimizers:
        o.step()

    return loss
        
    # Train_algo:
    # 1. run encoder
    # 2. get h_0 for decoder from encoder output
    # 3. decoder lstm one step
    # 4. attention context one step (encoder hidden state, decoder hidden state)
    # 5. run mlp one step
    # 6. goto step 3 till Max_Output_Size
