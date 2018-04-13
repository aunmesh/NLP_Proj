import importsys
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
import torch
from torch.autograd import Variable
from torch.optim import Adam
import pickle as pk
from  discriminator2 import Discriminator
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
# import Model
# import Model.as Constants
# from Model.Modules import Encoder, Generator, Discriminator
# from utils import check_cuda

PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<pad>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

epoch = 10
max_features = 10000
maxlen = 750
discriminator = Discriminator(10000, maxlen = 15)
batch_size = 128
criterion = torch.nn.CrossEntropyLoss()

# print('Loading data...')
# (x_train, y_train), (x_test, x_test[1]) = imdb.load_data(
#         num_words=max_features,
#         start_char=BOS,
#         oov_char=UNK,
#         index_from=EOS,
# )

# forward_dict = imdb.get_word_index()
# for key, value in forward_dict.items():
#     forward_dict[key] = value + EOS
# forward_dict[PAD_WORD] = PAD
# forward_dict[UNK_WORD] = UNK
# forward_dict[BOS_WORD] = BOS
# forward_dict[EOS_WORD] = EOS

# backward_dict = {}
# for key, value in forward_dict.items():
#     backward_dict[value] = key

classes = {
        "what":0,
        "how": 1,
        "why" :2,
        "when":3,
        "where":4,
        "who":5,
        "which":6
        }
# glove2word2vec('GloVe-1.2/glove.6B.50d.txt', 'GloVe-1.2/glove.6B.50d.txt.word2vec')
print("loading glove model")
glove = gensim.models.KeyedVectors.load_word2vec_format('GloVe-1.2/glove.6B.50d.txt.word2vec', binary= False )
print("loading complete")



def loadData(dataFile, labelFile):
    wordVector = []
    labels = []
    with open(dataFile, "r") as f:
        with open(labelFile, "r") as g:
            for line,label in zip(f,g):
                # wordVector = np.array(len(),750)
                num_words = 0
                for word in line.split():
                    if word in classes.keys():
                        continue
                    elif word in glove:
                        wordVector.append(glove[word])
                    else:
                        wordVector.append(glove["unk"])
                    num_words += 1
                    if(num_words >= 15):
                        break
                while num_words < 15:
                    wordVector.append(glove["pad"])
                    num_words += 1
                labels.append(int(label))

    wordVector = np.array(wordVector).reshape(-1,750)
    labels = np.array(labels)
    print("length of wordVector is ", wordVector.shape)
    return (wordVector, labels)

x_train = loadData("DiscTrain/ques_train.txt", "DiscTrain/ques_train_labels.txt")
x_test = loadData("DiscTrain/ques_test.txt", "DiscTrain/ques_test_labels.txt")

# x_train = sequence.pad_sequences(
#         x_train,
#         maxlen=maxlen,
#         padding='post',
#         truncating='post',
#         value=PAD,
#         )
# x_test = sequence.pad_sequences(
#         x_test,
#         maxlen=maxlen,
#         padding='post',
#         truncating='post',
#         value=PAD,
#         )



def get_batch(data, index, batch_size, testing=False):
    tensor = torch.from_numpy(data[index:index+batch_size]).type(torch.FloatTensor)
    input_data = Variable(tensor, volatile=testing, requires_grad=False)
    output_data = input_data
    return input_data, output_data

def get_batch_label(data, label, index, batch_size, testing=False):
    tensor = torch.from_numpy((data[index:index+batch_size]).reshape(-1,15,50))
    input_data = Variable(tensor, volatile=testing, requires_grad=False)
    label_tensor = torch.from_numpy(label[index:index+batch_size]).type(torch.FloatTensor)
    output_data = Variable(label_tensor, volatile=testing, requires_grad=False).type(torch.LongTensor)
    return input_data, output_data




# Borrow from https://github.com/ethanluoyc/pytorch-vae/blob/master/vae.py
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

d_opt = Adam(discriminator.parameters())



def train_discriminator(discriminator):
    # TODO: empirical Shannon entropy
    print_epoch = 9
    for epoch_index in range(epoch):
        for batch, index in enumerate(range(0, len(x_train[0]) - 1, batch_size)):
            # print("batch is ", batch, "index is ", index)
            discriminator.train()
            input_data, output_data = get_batch_label(
                                    x_train[0],
                                    x_train[1],
                                    index,
                                    batch_size
                                    )
            discriminator.zero_grad()
            output = discriminator(input_data, is_softmax=True, dont_pass_emb=True)
            loss = criterion(output, output_data)
            loss.backward()
            d_opt.step()
            if batch % 50 == 0:
                print("[Discriminator] Epoch {} batch {}'s loss: {}".format(
                    epoch_index,
                    batch,
                    loss.data[0],
                    ))
            # if True or (print_epoch == epoch_index and print_epoch):
        discriminator.eval()
        print_epoch = epoch_index + 1
        input_data, output_data = get_batch_label(x_test[0], x_test[1], 0, len(x_test[1]), testing=True)
        _, predicted = torch.max(discriminator(input_data, dont_pass_emb = True).data, 1)
        correct = (predicted == torch.from_numpy(x_test[1])).sum()
        print("[Discriminator] Test accuracy {} %".format(
            100 * correct / len(x_test[1])
            ))


outs = []
temp = []
offsets = []
gd = []
ctemp = []
coffsets = []
cgd = []
def testfn(discriminator):
    backward_dict = {}
    for key, value in classes.items():
        backward_dict[value] = key

    for batch, index in enumerate(range(0, len(x_test[0]) - 1, batch_size)):
        # print("batch is ", batch, "index is ", index)
        input_data, output_data = get_batch_label(
                x_test[0],
                x_test[1],
                index,
                batch_size)
        output = discriminator(input_data, is_softmax=True, dont_pass_emb=True)
        prob, predicted = torch.max(output.data, 1)
        for x in range(len(output)):
            if predicted[x]!=output_data.data[x]:
                # Error on this point!
                offset = batch*batch_size + x + 1
                offsets.append(int(offset))
                temp.append(np.asarray(torch.exp(output[x]).data))
                gd.append(np.asarray([int(predicted[x]), int(output_data.data[x])]))
                # outstr=(str(offset) + " error. Output :" + str(backward_dict[predicted[x]]) + " True:" + str(backward_dict[output_data.data[x]]) + " prob: " + str(torch.exp(output[x]).data) + "\n")
            else:
                # No error on this point!
                coffset = batch*batch_size + x + 1
                coffsets.append(int(coffset))
                ctemp.append(np.asarray(torch.exp(output[x]).data))
                cgd.append(np.asarray([int(predicted[x]), int(output_data.data[x])]))
        # outs.append(outstr)

model = train_discriminator(discriminator)
testfn(discriminator)

# temp = []
# for o in outs:
#     a = o.split('\n')
#     temp.append(a[1:8])
fo = open('err_offs.txt', 'w')
fv = open('err_vecs.txt', 'w')
fg = open('err_gd.txt', 'w')
np.savetxt('err_vecs.txt', temp)
np.savetxt('err_offs.txt', offsets)
np.savetxt('err_gd.txt', gd)


np.savetxt('correct_vecs.txt', ctemp)
np.savetxt('correct_offs.txt', coffsets)
np.savetxt('correct_gd.txt', cgd)
