import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import math


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
