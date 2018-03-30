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


class DecoderLSTM(nn.Module):
    def __init__(self, embedding, encoder_hidden_dim, decoder_hidden_dim, pad_idx, Wt_row_size, bos_token=0,
                 eos_token=1, decoder_layers=2, decoder_dropout_prob=0.3, max_size=15, target_vocab_size=28000):
        '''
            Input Args:
                embeddings          : nn.Embedding Layer containing source_vocab_size * embedding_size
                encoder_hidden_dim  : Hidden Dimension of The Encoder (To be used for attention while Decoding)
                decoder_hidden_dim  : Hidden Dimension of The Decoder (To be used for attention while Decoding)
                pad_idx             : pad_idx for the embedding
                Wt_row_size         : Number of Rows in first layer(Ws) of classification module.
                (Column size is decoder_hidden_dim + encoder_hidden_dim)
                bos_token           : Beginning of Sentence token
                eos_token           : End of Sentence token
                decoder_layers      : Number of Layers in the Decoder
                decoder_dropout_prob: Dropout Ratio to be applied in the decoder LSTM hidden layers
                max_size            : Max size of the output sequence
            Note:
                1. Note that the Encoder and Decoder Vocabularies are of different sizes.
                    This is checked using assert condition using a self.vocab_size.
                    Check its Value.
                2. Encoder_hidden_dim is same as 2 * Hidden_Unit_Size of Encoder_LSTM.
                    (2 is due to Bidirectional Forward Pass and further Concatentation)
        '''

        super(DecoderLSTM, self).__init__()
        self.target_vocab_size = target_vocab_size
        # Target Vaocabulary embedding
        self.embedding = embedding
        assert(self.embedding.weight.size(0) == self.target_vocab_size), "Embedding Passed is not of the required length {}".format(self.target_vocab_size)

        self.embed_dim = self.embedding.weight.size(1)

        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.pad_idx = pad_idx
        self.Wt_row_size = Ws_row_size
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.decoder_layers = decoder_layers
        self.decoder_dropout_prob = decoder_dropout_prob
        self.max_size = max_size

        self.decoder = nn.LSTM(self.embed_dim, self.decoder_hidden_dim, self.decoder_layers, dropout=self.decoder_dropout_prob)

        self.Wt = nn.Linear(self.decoder_hidden_dim + self.encoder_hidden_dim, self.Wt_row_size)
        self.Ws = nn.Linear(self.Wt_row_size, self.embed_dim)

        self.Attention = Attention_Context(self.encoder_dim, self.decoder_dim)

        def forward(self, encoded_input, input_data):
            '''
            Forward Function:
                Input Args:
                    ncoded_input   : Encoded Input Sentences (Size: [Batch, T, self.encoder_hidden_dim])
                    input_data      : Input Token List (Can be NoneType in not None then Size: [Batch, 1])
                Output:
                    Out_tokens      : Output_Softmax Value (Size: [Batch, max_size, self.vocab_size])
            '''














'''
class AttnDecoderRNN(nn.Module):
    def __init__(self, embed_dim, encoder_hidden_dim, hidden_dim,
                 vocab_size, bos_token=0, pad_idx=1, eos_token=2):
        """
        Initializes the RNN
        :param embed_dim: Dimension of the embeddings
        :param encoder_hidden_dim: Hidden dim of the encoder, for attention purposes
        :param hidden_dim: Hidden dim of the decoder
        :param vocab_size: Number of words in the vocab
        :param bos_token: To use during decoding (non teacher forcing mode))
        :param bos: beginning of sentence token
        :param unk: unknown token (not used)
        """
        self.bos_token = bos_token
        self.embed_dim = embed_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.eos_token = eos_token

        super(AttnDecoderRNN, self).__init__()

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=self.pad_idx)
        self.gru = nn.GRU(self.embed_dim + self.encoder_hidden_dim, self.hidden_dim)
        self.attn = Attention(self.encoder_hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.vocab_size)

        # Differs from the paper because I'm using the final forward and backward LSTM states
        self.init_hidden = nn.Linear(self.encoder_hidden_dim, self.hidden_dim)

    def _lstm_loop(self, state, embed, context, mask=None):
        """
        :param state: Current decoder state (batch_size, dec_dim)
        :param embed: Embedding size (batch_size, embed_dim)
        :param context: All the context from encoder (batch_size, source_l, enc_dim)
        :param mask: Mask of size (batch_size, source_l) with 1 if that token is valid in encoder,
                     0 otherwise.
        :return: out: (batch_size, vocab_size) distribution over labels
                 state: (batch_size, dec_dim) next state
                alpha: (batch_size, source_l) distribution over the encoded hidden states,
                       useful for debugging maybe
        """
        c_t, alpha = self.attn(state, context, mask)
        gru_inp = torch.cat((embed, c_t), 1).unsqueeze(0)

        state = self.gru(gru_inp, state.unsqueeze(0))[0].squeeze(0)
        out = self.out(state)

        return out, state, alpha
embeddin
    def _teacher_force(self, state, input_data, input_batches, context, mask):
        """
        Does teacher forcing for training
        :param state: (batch_size, dim) state size
        :param input_data: (t*batch_size) flattened array
        :param input_batches: Batch sizes for each timestep in input_data
        :param context: (T, batch_size, dim) of context
        :param mask: (T, batch_size) mask for context
        :return: Predictions (t*batch_size), exactly the same length as input_data
        """
        embeds = self.embedding(input_data)
        outputs = []
        for emb, batch_size in zip(packed_seq_iter((embeds, input_batches)),
                                   input_batches):

            out, state, alpha = self._lstm_loop(
                state[:batch_size],
                emb[:batch_size],
                context[:batch_size],
                mask[:batch_size],
            )
            outputs.append(out)
        return torch.cat(outputs)

    def _sample(self, state, context, mask, max_len=20):
        """
        Performs sampling
        """
        batch_size = state.size(0)

        toks = [const_row(self.bos_token, batch_size, volatile=True)]

        lens = torch.IntTensor(batch_size)
        if torch.cuda.is_available():
            lens = lens.cuda()

        for l in range(max_len + 1):  # +1 because of EOS
            out, state, alpha = self._lstm_loop(state, self.embedding(toks[-1]), context, mask)

            # Do argmax (since we're doing greedy decoding)
            toks.append(out.max(1)[1].squeeze(1))

            lens[(toks[-1].data == self.eos_token) & (lens == 0)] = l+1
            if all(lens):
                break
        lens[lens == 0] = max_len+1
        return torch.stack(toks, 0), lens

    def forward(self, h_cat, context, context_lens, input_data=None, max_len=20):
        """
        Does teacher forcing for training
        :param h_cat: (batch_size, d_enc*2) final state size
        :param inputs: PackedSequence (T*batch_size) of inputs
        :param context: (T, batch_size, dim) of context
        :param context_lens: (batch_size) Length of each batch
        :return:
        """
        state = self._init_hidden(h_cat)
        mask = rnn_mask(context_lens)

        if input_data is None:
            return self._sample(state, context, mask, max_len)

        if isinstance(input_data, PackedSequence):
            tf_out = self._teacher_force(state, input_data.data, input_data.batch_sizes, context, mask)
            return PackedSequence(tf_out, input_data.batch_sizes)

        # Otherwise, it's a normal torch tensor
        batch_size = input_data.size(1)
        T = input_data.size(0) - 1 # Last token is EOS

        tf_out = self._teacher_force(state, input_data[:T].view(-1), [batch_size] * T, context, mask)
        tf_out = tf_out.view(T, batch_size, -1)
        return tf_out

    def _init_hidden(self, h_dec):
        return F.tanh(self.init_hidden(h_dec))
'''
