from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from Attention import AttentionContext
from BiEself import EncoderLSTM
from DecoderNew import DecoderLSTM
from DecoderMLP import DecoderMLP
import re
from nltk.translate.bleu_score import corpus_bleu

import numpy as np

encoder_file = 'Encoder.pt'
decoder_file = 'DecoderLSTM.pt'
attention_file = 'Attention.pt'
mlp_file = 'DecoderMLP.pt'

target_vocab_size = 28442
input_size = 300
target_embed = nn.Embedding(target_vocab_size , input_size, padding_idx=-1).cuda()  # Make sure this is Glove Embeddings

src_embedding_file = '../Data/source_train_output.npy'
src_embedding = torch.from_numpy(np.load(src_embedding_file))


def load_data(filename):
    data = []
    file_handle = open(filename, 'r')
    for line in file_handle:
        line = line.strip()
        line = line.split()
        line = [int(i) for i in line]
        data.append(line)
    file_handle.close()
    return data


def pad(tensor, max_length):
    return torch.cat( ( tensor, torch.zeros((max_length)-tensor.size(0)).long() ) )


def sort_batch(src, tgt, src_lengths, batch_size):

    sorted_lengths, sorted_idx = src_lengths.sort()    # sort the length of sequence samples

    reverse_idx = torch.linspace(batch_size-1, 0, batch_size).long()

    sorted_lengths = sorted_lengths[reverse_idx]    # for descending order
    sorted_idx = sorted_idx[reverse_idx]

    src_batch_sorted = [src[i] for i in sorted_idx]                 # sorted in descending order
    tgt_batch_sorted = [tgt[i] for i in sorted_idx]

    return src_batch_sorted, tgt_batch_sorted, list(sorted_lengths)


def generate_batch(batch_size, src_data, tgt_data):

    src_lengths = torch.from_numpy(np.asarray([len(i) for i in src_data]))
    src_batch_sorted, tgt_batch_sorted, src_lengths = sort_batch(src_data, tgt_data, src_lengths, batch_size)

    max_src_length = max(src_lengths)
    src_batch_padded = []

    for sentence in src_batch_sorted:
        temp_arr = np.asarray(sentence)
        temp_tensor = torch.from_numpy(temp_arr)
        src_batch_padded.append(pad(temp_tensor, max_src_length))

    src_batch_padded = Variable(torch.stack(src_batch_padded), requires_grad=False).cuda()
    src_batch_packed = pack_padded_sequence(src_batch_padded, src_lengths, batch_first=True)

    return src_batch_packed, tgt_batch_sorted


def get_next_token_indices(softmax_vector):
    required_indices = torch.max(softmax_vector,1)[1]
    return required_indices


def get_embedding(indices):
    indices = Variable(indices.data.long() , requires_grad=False).cuda()
    return target_embed( indices ).cuda()


def Translator(source_batch_indices, batch_size):

    max_output_size = 20

    input_feature_dim = 300
    encoder_hidden_dim = 300
    decoder_hidden_dim = 600
    use_embedding = True
    source_vocab_size = 49909
    target_vocab_size = 28445
    mlp_hidden_dim = 100  # find an apt value from the original torch code

    model_encoder = EncoderLSTM(input_feature_dim, encoder_hidden_dim, use_embedding, source_vocab_size)
    model_encoder.embed.weight = nn.Parameter(src_embedding)

    model_attention = AttentionContext(2 * encoder_hidden_dim, decoder_hidden_dim)
    model_mlp = DecoderMLP(2 * encoder_hidden_dim, mlp_hidden_dim, decoder_hidden_dim, target_vocab_size)
    model_decoder = DecoderLSTM(encoder_hidden_dim, decoder_hidden_dim)

    model_encoder = model_encoder.cuda()
    model_decoder = model_decoder.cuda()
    model_attention = model_attention.cuda()
    model_mlp = model_mlp.cuda()

    temp_epoch = 15
    save_dir = "/new_data/gpu/aunmesh/Question_Generation/"
    encoder_file = save_dir + "Encoder_epoch_" + str(temp_epoch) + ".pt"
    decoder_file = save_dir + "DecoderLSTM_epoch_" + str(temp_epoch) + ".pt"
    mlp_file = save_dir + "DecoderMLP_epoch_" + str(temp_epoch) + ".pt"
    attention_file = save_dir + "Attention_epoch_" + str(temp_epoch) + ".pt"

    model_encoder.load_state_dict(torch.load(encoder_file))
    model_decoder.load_state_dict(torch.load(decoder_file))
    model_attention.load_state_dict(torch.load(attention_file))
    model_mlp.load_state_dict(torch.load(mlp_file))

    model_encoder.eval()
    model_decoder.eval()
    model_attention.eval()
    model_mlp.eval()
    num_iter = int(batch_size/64)

    for i in range(1,2):
        source_batch_indices = source_batch_indices
        encoder_output, encoder_output_len, encoder_states, mask_source = model_encoder(source_batch_indices)

        ################################################################
        encoder_hidden_states = encoder_states[0]  # [batch, num_layers * num_directions, hidden_size]
        encoder_cell_states = encoder_states[1]

        dh_0_1 = torch.cat((encoder_hidden_states[0], encoder_hidden_states[1]), 1)
        dh_0_2 = torch.cat((encoder_hidden_states[2], encoder_hidden_states[3]), 1)
        dh_0 = torch.stack((dh_0_1, dh_0_2), 0)

        dc_0_1 = torch.cat((encoder_cell_states[0], encoder_cell_states[1]), 1)
        dc_0_2 = torch.cat((encoder_cell_states[2], encoder_cell_states[3]), 1)
        dc_0 = torch.stack((dc_0_1, dc_0_2), 0)

        decoder_prev_state = (dh_0, dc_0)

        sos_indices_batch = Variable(torch.ones(batch_size, 1))  # One is he index for SOS tag
        prev_token = get_embedding(sos_indices_batch[:, 0])
        prev_token = prev_token.unsqueeze(1)
        ################################################################

        decoded_indices_sub = Variable(torch.ones((batch_size,1))).cuda()

        for step in range(1, max_output_size):
            decoder_next_output, decoder_next_state = model_decoder(prev_token, decoder_prev_state)
    	    context_vec, __ = model_attention(encoder_output, decoder_next_output, mask_source)

            # softmax_vector : [ Batch_Size , Target_Vocab_Size ]
            softmax_vector = model_mlp(decoder_next_output, context_vec)
            next_token_indices = get_next_token_indices(softmax_vector)
            next_token_indices = Variable(next_token_indices.data.float() , requires_grad=False).cuda()
            next_token = get_embedding(next_token_indices)

            #print(decoded_indices_sub)
            #print(next_token_indices)

            decoded_indices_sub = torch.cat((decoded_indices_sub, next_token_indices.unsqueeze(1)), 1)

            prev_token = next_token.unsqueeze(1)
            decoder_prev_state = decoder_next_state
    
        if(i>1):
            decoded_indices = torch.cat((decoded_indices, decoded_indices_sub),0)
        else:
            decoded_indices = decoded_indices_sub
	
    return decoded_indices


if __name__ == '__main__':

    src_data = load_data('../Data/src_data_test.txt')
    total_size = len(src_data)
    batch_size = 64
    num_iter = int(total_size/batch_size)

    target_questions = []
    tgt_test_file = '../Data/tgt-test.txt'
    token_pattern = re.compile(r"[a-zA-Z]+")
    with open(tgt_test_file, "r") as input_file:
        for i, line in enumerate(input_file):
            words = token_pattern.findall(line)
            target_questions.append([words])

    candidates = []
    references = []
    for i in range(1,4):#num_iter):
        print(i)
        src = src_data[(i-1)*64: i*64]
	tgt = target_questions[(i-1)*64: i*64]
        src_batch_indices, target_questions_sorted = generate_batch(batch_size, src, tgt)

    	decoded_indices = Translator(src_batch_indices, batch_size)
    	references += target_questions_sorted

    	tgt_vocab_file = '../Data/tgt_vocab.txt'
    	f = open(tgt_vocab_file, 'r')
    	tgt_vocab = []
    	for line in f:
            tgt_vocab.append(line.strip())

    	decoded_words = []
    	for sentence in decoded_indices:
            temp = []
            for word in sentence:
                temp.append(tgt_vocab[int(word)])

            decoded_words.append(temp)

    	candidates += decoded_words

    bleu1 = corpus_bleu(references, candidates, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, candidates, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, candidates, weights=(0.25, 0.25, 0.25, 0.25))
    for c in candidates:
        print(" ".join(c))
        print("\n")
    print ("Test_Batch_Size: ", len(candidates))
    print("Bleu-1 Score on Corpus: ", bleu1)
    print("Bleu-2 Score on Corpus: ", bleu2)
    print("Bleu-3 Score on Corpus: ", bleu3)
    print("Bleu-4 Score on Corpus: ", bleu4)









