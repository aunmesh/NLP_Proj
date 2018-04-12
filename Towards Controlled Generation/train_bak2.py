import importsys
import numpy as np
from keras.preprocessing import sequence
from keras.datasets import imdb
import torch
from torch.autograd import Variable
from torch.optim import Adam
import pickle as pk
from  discriminator import Discriminator
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

epoch = 20
max_features = 10000
maxlen = 20
discriminator = Discriminator(10000)
batch_size = 64
criterion = torch.nn.CrossEntropyLoss()

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=max_features,
        start_char=BOS,
        oov_char=UNK,
        index_from=EOS,
)

forward_dict = imdb.get_word_index()
for key, value in forward_dict.items():
    forward_dict[key] = value + EOS
forward_dict[PAD_WORD] = PAD
forward_dict[UNK_WORD] = UNK
forward_dict[BOS_WORD] = BOS
forward_dict[EOS_WORD] = EOS

backward_dict = {}
for key, value in forward_dict.items():
    backward_dict[value] = key

x_train = sequence.pad_sequences(
        x_train,
        maxlen=maxlen,
        padding='post',
        truncating='post',
        value=PAD,
        )
x_test = sequence.pad_sequences(
        x_test,
        maxlen=maxlen,
        padding='post',
        truncating='post',
        value=PAD,
        )

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    f = open("glove_model.pkl","wb")
    pk.dump(model,f)
    return model

def loadModel(flag = False):
    print("loadModel start")
    if flag:
        model = loadGloveModel("GloVe-1.2/glove.6B.100d.txt")
    else:
        f = open("glove_model.pkl", 'rb')
        model  = pk.load(f)
    print("loadModel end")

    return model



def get_batch(data, index, batch_size, testing=False):
    tensor = torch.from_numpy(data[index:index+batch_size]).type(torch.LongTensor)
    input_data = Variable(tensor, volatile=testing, requires_grad=False)
    output_data = input_data
    return input_data, output_data

def get_batch_label(data, label, index, batch_size, testing=False):
    tensor = torch.from_numpy(data[index:index+batch_size]).type(torch.LongTensor)
    input_data = Variable(tensor, volatile=testing, requires_grad=False)
    label_tensor = torch.from_numpy(label[index:index+batch_size]).type(torch.LongTensor)
    output_data = Variable(label_tensor, volatile=testing, requires_grad=False)
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
        for batch, index in enumerate(range(0, len(x_train) - 1, batch_size)):
            discriminator.train()
            input_data, output_data = get_batch_label(
                    x_train,
                    y_train,
                    index,
                    batch_size
                    )

            discriminator.zero_grad()

            output = discriminator(input_data)
            loss = criterion(output, output_data)
            loss.backward()
            d_opt.step()

            if batch % 25 == 0:
                print("[Discriminator] Epoch {} batch {}'s loss: {}".format(
                    epoch_index,
                    batch,
                    loss.data[0],
                    ))
            # if True or (print_epoch == epoch_index and print_epoch):
        discriminator.eval()
        print_epoch = epoch_index + 1
        input_data, output_data = get_batch_label(x_test, y_test, 0, len(y_test), testing=True)
        _, predicted = torch.max(discriminator(input_data).data, 1)
        correct = (predicted == torch.from_numpy(y_test)).sum()
        print("[Discriminator] Test accuracy {} %".format(
            100 * correct / len(y_test)
            ))


# glove = loadModel(True)
model = train_discriminator(discriminator)
