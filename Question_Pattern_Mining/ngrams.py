import nltk
from nltk import ngrams
import string
from nltk.corpus import stopwords

swl = set(stopwords.words('english'))

#Clears a sentence of Punctuation Chars and converts the sentence to lower chars
def ClearPunct(s):
    return s.translate(None, string.punctuation).lower()

#s is the input sentence, returns the number of content words
def Contentify2(s):
    s = ClearPunct(s)
    text = nltk.tokenize.word_tokenize(s)
    tags = nltk.pos_tag(text)

    # NN is noun, JJ is adjectives, RB is adverbs , VB is verbs
    C_words = ['NN','JJ','RB', 'VB']
    count = 0
    
    for x in tags:
        if x[1][0] + x[1][1] in C_words and x[0] not in swl:
            print(x)
            count += 1
    return count


# Generates 
def Ngrams(sentence, n):
    s = ClearPunct(sentence)
    Fgrams = ngrams(s.split(),n)
    res = []
    for grams in Fgrams:
        res.append(grams)
    return res