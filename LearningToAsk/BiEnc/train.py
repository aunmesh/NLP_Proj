from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from pytorch_misc import rnn_mask, packed_seq_iter, seq_lengths_from_pad, const_row
from torch import optim
from Attention import AttentionContext
from BiEself import EncoderLSTM
from DecoderNew import DecoderLSTM
from DecoderMLP import DecoderMLP
from train_batch import train_batch


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
    source_vocab_size = 45000
    mlp_hidden_dim = 100                                # find an apt value from the original torch code

    encoder = EncoderLSTM(input_feature_dim, encoder_hidden_dim, use_embedding, source_vocab_size)
    attention = AttentionContext(encoder_hidden_dim, decoder_hidden_dim)
    mlp = DecoderMLP(encoder_hidden_dim, mlp_hidden_dim)
    decoder = DecoderLSTM(encoder_hidden_dim, mlp_hidden_dim)

    criterion = nn.CrossEntropyLoss()                   # WRONG

    encoder_learning_rate = 0.01
    decoder_learning_rate = 0.01

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=encoder_learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=decoder_learning_rate)

    dataset_size = 50000                                # Not the actual value
    batch_size = 64

    train_loader = None                                 # define a dataset loader

    max_output_size = 20
    epochs = 10
    num_iterations = dataset_size / batch_size

    for e in range(0, epochs):
        for it, batch in enumerate(train_loader):
            print(e,it,batch)
            loss = train_batch(batch, encoder, decoder, attention, mlp, criterion,
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