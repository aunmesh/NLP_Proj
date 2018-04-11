from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer
import json
import numpy as np
from sets import Set
import re


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


def main():

    word2embedding = {}
    # token_pattern = re.compile(r"\w+")
    token_pattern = re.compile(r"[a-zA-Z]+")

    with open("./glove/glove.6B.300d.txt", "r") as input_file:
        for line in input_file:
            line = line.split()
            word2embedding[line[0]] = np.asarray(map(float, line[1 : ]))
            dimension = len(line) - 1

    words = Set()
    with open("src-train.txt", "r") as input_file:
        for line in input_file:
            tokens = token_pattern.findall(line)
            for token in tokens:
                words.add(token)

    embedding = np.random.uniform(low=-1.0 / 3, high=1.0 / 3, size=(len(words), dimension))
    embedding = np.asarray(embedding, dtype=np.float32)

    f = open("src_vocab.txt", "w")
    unknown_count = 0
    for i, word in enumerate(words):
        if word in TRANSLATE:
            word = TRANSLATE[word]
        done = False
        for w in (word, word.upper(), word.lower()):
            if w in word2embedding:
                f.write(w)
                f.write('\n')
                embedding[i] = word2embedding[w]
                done = True
                break
        if not done:
            print("Unknown word: %s" % (word, ))
            unknown_count += 1

    f.close()

    np.save("source_train_output", embedding)
    print("Total unknown: %d" % (unknown_count, ))

    words = Set()
    with open("tgt-train.txt", "r") as input_file:
        for line in input_file:
            tokens = token_pattern.findall(line)
            for token in tokens:
                words.add(token)

    embedding = np.random.uniform(low=-1.0 / 3, high=1.0 / 3, size=(len(words), dimension))
    embedding = np.asarray(embedding, dtype=np.float32)

    f = open("tgt_vocab.txt", "w")

    unknown_count = 0
    for i, word in enumerate(words):
        if word in TRANSLATE:
            word = TRANSLATE[word]
        done = False
        for w in (word, word.upper(), word.lower()):
            if w in word2embedding:
                f.write(w)
                f.write('\n')
                embedding[i] = word2embedding[w]
                done = True
                break
        if not done:
            print("Unknown word: %s" % (word,))
            unknown_count += 1

    f.close()
    np.save("source_target_output", embedding)
    print("Total unknown: %d" % (unknown_count,))

    src_vocab = {}
    f = open("src_vocab.txt", "r")
    for i, line in enumerate(f):
        src_vocab[line.strip()] = i
    f.close()

    src_index_train = []

    with open("src-train.txt", "r") as input_file:
        for i, line in enumerate(input_file):
            words = token_pattern.findall(line)
            temp = []
            for j, word in enumerate(words):
                try:
                    temp.append(str(src_vocab[word]))
                except:
                    temp.append(str(-1))
            src_index_train.append(temp)

    f = open("src_data_train.txt", "w")
    for sentence in src_index_train:
        line = " ".join(sentence)
        f.write(line)
        f.write('\n')
    f.close()

    tgt_vocab = {}
    f = open("tgt_vocab.txt", "r")
    for i, line in enumerate(f):
        tgt_vocab[line.strip()] = i
    f.close()

    tgt_index_train = []

    with open("tgt-train.txt", "r") as input_file:
        for i, line in enumerate(input_file):
            words = token_pattern.findall(line)
            temp = []
            for j, word in enumerate(words):
                try:
                    temp.append(str(tgt_vocab[word]))
                except:
                    temp.append(str(-1))
            tgt_index_train.append(temp)

    f = open("tgt_data_train.txt", "w")
    for sentence in tgt_index_train:
        line = " ".join(sentence)
        f.write(line)
        f.write('\n')
    f.close()


if __name__ == "__main__":
    main()