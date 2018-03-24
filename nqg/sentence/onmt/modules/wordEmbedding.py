import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import math

# Parameters:
#   * `vocabSize` - size of the vocabulary
#   * `vecSize` - size of the embedding
#   * `preTrainined` - path to a pretrained vector file
#   * `fix` - keep the weights of the embeddings fixed.

class WordEmbedding(nn.Module):
	def __init__(self, vocabSize, vecSize, preTrained, fix):
		super(WordEmbedding, self).__init__()
		self.vocabSize = vocabSize
		self.model = nn.Sequential()
		self.net = nn.Embedding(vocabSize, vecSize, padding_idx=1)
		self.model.add_module("WENet", self.net)

		if preTrained and len(preTrained) > 0:
			temp = torch.load(preTrained)
			# self.net.load_state_dict(temp)
			self.model.WENet.load_state_dict(temp)

			self.fix = fix
			if self.fix:
				self.net.zero_grad()

	def postParametersInitialization():
        # This value is taken from onmt/Constants
		PAD = 1
        # Careful: lua tensor indices start from 1
		self.net.weight.data[PAD-1] = torch.FloatTensor(self.net.weight[PAD-1].size()).zero_()
