from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re

TRANSLATE = {
    "-lsb-": "[",
    "-rsb-": "]",
    "-lrb-": "(",
    "-rrb-": ")",
    "-lcb-": "{",
    "-rcb-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
}


def main():

    token_pattern = re.compile(r"[a-zA-Z]+")
    src_vocab = {}
    f = open("src_vocab.txt", "r")
    for i, line in enumerate(f):
        src_vocab[line.strip()] = i
    f.close()

    src_index_test = []

    with open("src-test.txt", "r") as input_file:
        for i, line in enumerate(input_file):
            words = token_pattern.findall(line)
            temp = [str(1)]          # represents <SOS> tag
            for j, word in enumerate(words):
                try:
                    temp.append(str(src_vocab[word]))
                except:
                    temp.append(str(0))  # represents <UNK> tag
            temp.append(str(2))  # represents <EOS> tag
            src_index_test.append(temp)

    f = open("src_data_test.txt", "w")
    for sentence in src_index_test:
        line = " ".join(sentence)
        f.write(line)
        f.write('\n')
    f.close()

if __name__ == "__main__":
    main()
