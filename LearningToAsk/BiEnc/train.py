from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence, pack_padded_sequence
from pytorch_misc import rnn_mask, packed_seq_iter, seq_lengths_from_pad, const_row
from torch import optim
from Attention import AttentionContext
from BiEself import EncoderLSTM
from DecoderNew import DecoderLSTM
from DecoderMLP import DecoderMLP
from train_batch import train_batch
from torch.autograd import Variable

import numpy as np




def load_data(filename):
    data = []
    file_handle = open(filename, 'r')
    for line in file_handle:
        line = line.strip()
        line = line.split()
        line = [int(i) + 1 for i in line]   #Temporary fix for indices
        data.append(line)
    file_handle.close()
    return data

src_data = load_data('../Data/src_data_train.txt')
tgt_data = load_data('../Data/tgt_data_train.txt')
data_size = len(src_data)
batch_size = 100

src_embedding_file = '../Data/source_train_output.npy'
src_embedding = torch.from_numpy(np.load(src_embedding_file))


def pad(tensor, max_length):
    return torch.cat( ( tensor, torch.zeros((max_length)-tensor.size(0)).long() ) )


def sort_batch(src, tgt, src_lengths, tgt_lengths):

    sorted_lengths, sorted_idx = src_lengths.sort()    # sort the length of sequence samples

    reverse_idx = torch.linspace(batch_size-1, 0, batch_size).long()

    sorted_lengths = sorted_lengths[reverse_idx]    # for descending order
    sorted_idx = sorted_idx[reverse_idx]

    src_batch_sorted = [src[i] for i in sorted_idx]                 # sorted in descending order
    tgt_batch_sorted = [tgt[i] for i in sorted_idx]
    tgt_lengths = [tgt_lengths[i] for i in sorted_idx]


    return src_batch_sorted, tgt_batch_sorted, list(sorted_lengths), tgt_lengths


def generate_batch(batch_size):
    indices = np.random.randint(0,data_size, batch_size)

    src_batch_raw = [src_data[i] for i in indices]
    tgt_batch_raw = [tgt_data[i] for i in indices]
    src_lengths = torch.from_numpy(np.asarray([len(i) for i in src_batch_raw]))
    tgt_lengths = [len(i) for i in tgt_batch_raw]

    src_batch_sorted, tgt_batch_sorted, src_lengths, tgt_lengths = sort_batch(src_batch_raw, tgt_batch_raw, src_lengths, tgt_lengths)

    max_src_length = max(src_lengths)
    src_batch_padded = []

    for sentence in src_batch_sorted:

        temp_arr = np.asarray(sentence)
        temp_tensor = torch.from_numpy(temp_arr)

        src_batch_padded.append(pad(temp_tensor, max_src_length))

    src_batch_padded = Variable( torch.stack(src_batch_padded), requires_grad=False)

    src_batch_packed = pack_padded_sequence(src_batch_padded, src_lengths, batch_first=True)

    max_tgt_length = max(tgt_lengths)
    tgt_batch_padded = []
    for question in tgt_batch_sorted:
        temp_arr = np.asarray(question)
        temp_tensor = torch.from_numpy(temp_arr)


        tgt_batch_padded.append(pad(temp_tensor, max_tgt_length))

    tgt_batch_padded = Variable(torch.stack(tgt_batch_padded), requires_grad=False)

    return src_batch_packed, (tgt_batch_padded, tgt_lengths)


def train_loader(batch_size, num_iterations):
    for num in range(num_iterations):
        yield generate_batch(batch_size)


def main():

    """
        input_size: feature size of the input
        hidden_size: size of the hidden state of the LSTM
        use_embedding: true if we have to use embedding
        source_vocab_size: size of the vocabulary (needed only if use_embedding is true)
        pad_idx: pad_indices(needed only if use_embedding is true)
    """

    input_feature_dim = 300
    encoder_hidden_dim = 300
    decoder_hidden_dim = 600
    use_embedding = True
    source_vocab_size = 49906
    target_vocab_size = 28442
    mlp_hidden_dim = 100                                # find an apt value from the original torch code

    encoder = EncoderLSTM(input_feature_dim, encoder_hidden_dim, use_embedding, source_vocab_size)
    encoder.embed.weight = nn.Parameter(src_embedding)

    attention = AttentionContext(2*encoder_hidden_dim, decoder_hidden_dim)
    mlp = DecoderMLP(2*encoder_hidden_dim, mlp_hidden_dim, decoder_hidden_dim, target_vocab_size)
    decoder = DecoderLSTM(encoder_hidden_dim, decoder_hidden_dim)

    encoder_learning_rate = 0.01
    decoder_learning_rate = 0.01

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=encoder_learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=decoder_learning_rate)

    max_output_size = 20
    epochs = 10

    num_iteration = int(data_size / batch_size)


    for e in range(0, epochs):
        for it, batch in enumerate(train_loader(batch_size, num_iteration)):
            print(e, it)
            loss = train_batch(batch, encoder, decoder, attention, mlp,
                               (encoder_optimizer, decoder_optimizer), max_output_size)

            if it % 10 == 0:
                print(e, it, loss)

    # Describe loss function
    # Describe optimizer
    # Define dataset iterator which, for a given batch size, gives source passage and ground truth
    # Define Optimizer
    # Define Max_Output_Size

    # num_iterations = DatasetSize/ Batch Size

    # [Outer Loop] For total no. of epochs
    #   [Inner Loop] For num_iterations
    #     Run Train_Algo
    #     CAlculate_Loss
    #     Clear model gradients
    #     BackProp
    #     Call optimizer_step

    # Train_algo:
    # 1. run encoder
    # 2. get h_0 for decoder from encoder states
    # 3. decoder lstm one step
    # 4. attention context one step (encoder hidden state, decoder hidden state)
    # 5. run mlp one step
    # 6. goto step 3 till Max_Output_Size


if __name__ == '__main__':

    main()














'''
def random_data(batch_size):
    sentences = np.random.randint(1,9,10*batch_size, dtype=int)
    sentences = sentences.reshape(batch_size,10)

    questions =  np.random.randint(1,5,6*batch_size, dtype=int)
    questions =  questions.reshape(batch_size,6)
    #torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False)
    temp = ( Variable( torch.from_numpy(sentences), requires_grad=False),Variable( torch.from_numpy(questions), requires_grad=False) )

    lengths_sentences = [10] * batch_size
    lengths_questions = [6] * batch_size

    temp1 = pack_padded_sequence( temp[0], lengths_sentences, batch_first=True)
    temp2 = pack_padded_sequence( temp[1], lengths_questions, batch_first=True)

    return (temp1, temp2)

'''
