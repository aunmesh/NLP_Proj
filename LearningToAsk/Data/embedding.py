from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
import json

from six.moves import xrange
from six.moves import zip
from six.moves import cPickle


import numpy as np


TRANSLATE = {
    "-lsb-" : "[",
    "-rsb-" : "]",
    "-lrb-" : "(",
    "-rrb-" : ")",
    "-lcb-" : "{",
    "-rcb-" : "}",
    "-LSB-" : "[",
    "-RSB-" : "]",
    "-LRB-" : "(",
    "-RRB-" : ")",
    "-LCB-" : "{",
    "-RCB-" : "}",
}


# def parse_args(description = "I am lazy"):
#     import argparse
#     parser = argparse.ArgumentParser(description = description)
#     parser.add_argument("--embedding", type = str, default = "./glove/glove.6B.100d.txt", required = True)
#     parser.add_argument("--dict", type = str, required = True)
#     parser.add_argument("--output", type = str, required = True)
#     parser.add_argument("--seed", type = int, default = 19941023)
#     args = parser.parse_args()
#     np.random.seed(args.seed)
#     return args


def main():
    # args = parse_args()

    word2embedding = {}
    dimension = None
    with open("./glove/glove.6B.100d.txt", "r") as input_file:
        for line in input_file:
            line = line.split()
            word2embedding[line[0]] = np.asarray(map(float, line[1 : ]))
            dimension = len(line) - 1

    with open("src-train.txt", "r") as input_file:
        words = [ line.split()[0] for line in input_file ]

    embedding = np.random.uniform(low = -1.0 / 3, high = 1.0 / 3, size = (len(words), dimension))
    embedding = np.asarray(embedding, dtype = np.float32)
    unknown_count = 0
    for i, word in enumerate(words):
        if word in TRANSLATE:
            word = TRANSLATE[word]
        done = False
        for w in (word, word.upper(), word.lower()):
            if w in word2embedding:
                embedding[i] = word2embedding[w]
                done = True
                break
        if not done:
            print("Unknown word: %s" % (word, ))
            unknown_count += 1

    # vectorizer = CountVectorizer(token_pattern='(?u)\\b\\w+\\b')
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(words)
    dict_vocab = json.dumps(vectorizer.vocabulary_)
    f = open("data_train_src.json", "w")
    f.write(dict_vocab)
    f.close()
    np.save("source_train_output", embedding)
    print("Total unknown: %d" % (unknown_count, ))


    with open("tgt-train.txt", "r") as input_file:
        words = [line.split()[0] for line in input_file]

    embedding = np.random.uniform(low=-1.0 / 3, high=1.0 / 3, size=(len(words), dimension))
    embedding = np.asarray(embedding, dtype=np.float32)
    unknown_count = 0
    for i, word in enumerate(words):
        if word in TRANSLATE:
            word = TRANSLATE[word]
        done = False
        for w in (word, word.upper(), word.lower()):
            if w in word2embedding:
                embedding[i] = word2embedding[w]
                done = True
                break
        if not done:
            print("Unknown word: %s" % (word,))
            unknown_count += 1

    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(words)

    dict_vocab = json.dumps(vectorizer.vocabulary_)
    f = open("data_train_tgt.json", "w")
    f.write(dict_vocab)
    f.close()

    np.save("source_target_output", embedding)
    print("Total unknown: %d" % (unknown_count,))



    src_index_train = []

    src_vocab = json.load(open('data_train_src.json'))

    with open("src-train.txt", "r") as input_file:
        for i, line in enumerate(input_file):
            words = line.split()[0]
            temp = []
            for word in words:
                if len(word) > 1:
                    temp.append(src_vocab[word])
            src_index_train.append(temp)
    np.save("src_index_train" , np.array(src_index_train))

    tgt_index_train = []
    tgt_vocab = json.load(open('data_train_tgt.json'))

    with open("tgt-train.txt", "r") as input_file:
        for i, line in enumerate(input_file):
            words = line.split()[0]
            temp = []
            for word in words:
                if len(word) > 1:
                    temp.append(tgt_vocab[word])
            tgt_index_train.append(temp)
    np.save("tgt_index_train" , np.array(tgt_index_train))

if __name__ == "__main__":
    main()