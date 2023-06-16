from collections import Counter
import codecs
import itertools
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence
def create_maps(words, min_word_freq=1, min_char_freq=1,*word_dicttionary):
    """
    Creates word, char, tag maps.

    :param words: word sequences
    :param tags: tag sequences
    :param min_word_freq: words that occur fewer times than this threshold are binned as <unk>s
    :param min_char_freq: characters that occur fewer times than this threshold are binned as <unk>s
    :return: word, char, tag maps
    """
    word_freq = Counter()
    char_freq = Counter()
    for w in words:
        word_freq.update(w)
        char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), w)))

    dictionary = {}
    for d in words:
            if d not in dictionary:
                dictionary[d] = len(dictionary)
        

    word_map = {k: v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] > min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c in char_freq.keys() if char_freq[c] > min_char_freq])}

    #dictionary['<pad>'] = -1
    #dictionary['<end>'] = len(dictionary)
    #dictionary['<unk>'] = len(dictionary)
    #char_map['<pad>'] = -1
    #char_map['<end>'] = len(char_map)
    #char_map['<unk>'] = len(char_map)

    return dictionary, char_map

def make_dictionary(wordlist):
    dictionary = {}
    for d in wordlist:
            if d not in dictionary:
                dictionary[d] = len(dictionary)
    return dictionary


