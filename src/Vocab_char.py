##################****************##############
###Author - Koustava Goswami################
###########****************##############

from tqdm import tqdm
import json
import numpy as np
from collections import Counter
from torch.utils import data
# from torchtext import datasets
import torch
from torch.utils.data import Dataset as dataset
# import pandas as pd
from tqdm import tqdm
from keras_preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
import string

from torch.nn.utils.rnn import pad_sequence


def load_data(args):
    text = []
    sentence_length = []
    word_vocab = []
    target_texts = list()
    input_characters = set()
    f = open(args.data, "r")

    df = pd.read_fwf(args.data, header=None)
    measurer = np.vectorize(len)
    # print (df.col1.map(lambda x: len(x)).max())
    maxCharacterSize = measurer(df.values.astype(str)).max(axis=0)
    # print(maxCharacterSize)
    text = df.values.tolist()
    flatText = [item for sublist in text for item in sublist]
    max_sentence_length = max([len(txt) for txt in flatText])
    for txt in flatText:
        # print(txt)
        sentence_length.append(len(txt.split()))
        word_vocab.append(word_tokenize(txt))
        for char in txt:
            if char not in input_characters:
                input_characters.add(char)
    flatTokenList = [item for sublist in word_vocab for item in sublist]
    max_word_length = max([len(x) for x in flatTokenList])
    maxSentenceSize = max(sentence_length)
    input_characters.add('<S>')
    input_characters.add('</S>')
    input_characters.add('<UNK>')
    numberOfCharacter = len(input_characters)
    for i in flatText:
        target_text = '<S> ' + i + ' </S>'
        target_texts.append(target_text)
    return target_texts, maxCharacterSize, maxSentenceSize, input_characters, numberOfCharacter, max_word_length, flatTokenList


def prepare_sequence(seq, to_ix, char2id):
    idx_final = []
    dicti = {}
    for i in range(len(seq)):
        idx = []
        for w in seq[i]:
            idxs = to_ix[w]
            # ida = prepare_character_sequence(w,char2id)
            idx.append(idxs)
            # idx.append(ida)
        idx_final.append(idx)
        # dicti.update({idx_final : seq})
    # idxs = [to_ix[w] for w in seq]
    return idx_final  # torch.tensor(idxs, dtype=torch.long)


def prepare_character_sequence(seq, to_ix, char2id):
    idx_final = []

    for i in range(len(seq)):
        dicti = []
        for w in seq[i]:
            for c in w:
                idx = char2id[c]
                dicti.append(idx)
            dicti.append(1)
        idx_final.append(dicti)
    return idx_final  # torch.tensor(idxs, dtype=torch.long)


def prepare_character_sequence1(seq, to_ix, char2id):
    idx_final = []

    for i in range(len(seq)):
        chard = []
        for w in seq[i]:
            dicti = []
            for c in w:
                idx = char2id[c]
                dicti.append(idx)
            dicti.append(1)
            chard.append(dicti)
        idx_final.append(chard)
    return idx_final  # torch.tensor(idxs, dtype=torch.long)


# Reading the embeddings from fasttext and populate embeddings for the words which are not there in fasttext
# def pretrained_embedding_matrix(file_name):
#     embeddings_index = dict()
#     words=[]
#     for line in file_name:
#         values = line.split()
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#         words.append(word)
#     file_name.close()
#     print('Loaded %s word vectors.' % len(embeddings_index))

#     inputs,raw_inputs, raw_seqs,targets = prepare_sentence('/home/kougos/data/language_detection_dataset/supervised_code_model/fasttext_k-means/german_dataset.csv')

#     x = set(raw_inputs)

#     x= list(x)

#     x.insert(0,"<pad>")


#     matrix_len = len(x)
#     weights_matrix = torch.zeros((matrix_len, 100)).long()
#     words_found = 0

#     for i, word in enumerate(x):
#         if(i==0):
#             continue
#         else:
#             try: 
#                 weights_matrix[i] = torch.FloatTensor(embeddings_index[word])
#                 words_found += 1
#             except KeyError:
#                 weights_matrix[i] = torch.randn(100, dtype=torch.float)#torch.random.normal(scale=0.6, size=(100, ))
#     return weights_matrix,x


def read_data(file_path, stemming_arabic=False, new_preprocess=False):
    labels = ["HIN", "MAG", "ENG"]
    inputs, targets, raw_inputs, raw_seqs = [], [], [], []
    line_count = 0
    p_bar = tqdm(total=322306)
    with open(file_path, "r", encoding="utf-8") as file_in:
        input_seq, target_seq, raw_seq = [], [], []
        # file_in = file_in[0:25000]
        for line in file_in:
            p_bar.update(1)
            line_count += 1
            # if line_count>=2000:
            #     break
            line = line.replace('\n', '')
            # arr = line.split(',')
            arr = line.split('\t')
            try:
                lanid, sent = arr[1], arr[0]
                lanid = lanid.strip()
                sent = sent.strip()
            except:
                print("line_count = ")
            if line_count % 10000 == 0:
                print("L:%s T:%s" % (lanid, sent))
            if lanid not in labels:
                continue
            # table = str.maketrans('', '', string.punctuation)
            if sent != " text":
                tokens = word_tokenize(sent)
                for token in tokens:
                    # word_vocab = [token]


                    # for w in range(len(word_vocab)):
                    if token != '``' and token != "''":
                        # and word_vocab[w] not in string.punctuation
                        input_seq.append(token.lower())
                        raw_seq.append(token.lower())
                        raw_inputs.append(token.lower())
                        target_seq.append(lanid)
                    if len(input_seq) > 0:
                        if input_seq not in inputs:
                            # input_seq = [w.translate(table) for w in input_seq]
                            inputs.append(input_seq)
                        if raw_seq not in raw_seqs:
                            # raw_seq = [w.translate(table) for w in raw_seq]
                            raw_seqs.append(raw_seq)
                            targets.append(target_seq)
                            if len(raw_seqs) != len(targets):
                                print("Mismatch")
                        input_seq, target_seq, raw_seq = [], [], []
    #print(f"{len(inputs)=}")
    #print(f"{len(raw_inputs)=}")
    #print(f"{len(raw_seqs)=}")
    #print(f"{len(targets)=}")
    return inputs, raw_inputs, raw_seqs, targets


def prepare_sentence(train_file):
    inputs, raw_inputs, raw_seqs, targets = read_data(train_file)

    return inputs, raw_inputs, raw_seqs, targets


##End Fasttext embedding loading part########

def prepare_dataset(supv_unspv, train_test, train_file, test_file, batch_size, eval_batch_size, Overwrite_label,
                    index_label_dic):
    all_inputs, all_raw_inputs, all_raw_seqs, targets = read_data(train_file)

    ###Added for test dataset
    test_inputs, all_raw_inputs_test, test_raw_seqs, test_targets = read_data(test_file)
    ###Addition end

    # file_name = open('/home/kougos/data/language_detection_dataset/fasttext_k-means/embedding_final.vec', 'r', encoding='utf8', errors='ignore')
    # weights_matrix, raw_inputs = pretrained_embedding_matrix(file_name)
    # print(all_inputs)

    num_train = int(len(all_inputs) * 0.95)
    num_valid = len(all_inputs) - num_train

    # if (train_test == "train" and supv_unspv == "supv"):
    train_inputs, train_raw_seqs, train_targets = all_inputs[:num_train], all_raw_seqs[:num_train], targets[:num_train]
    # else:
    #     train_inputs, train_raw_seqs,train_targets = all_inputs,all_raw_seqs,targets
    train_raw_inputs = []

    for i in tqdm(range(len(train_raw_seqs))):
        train_raw_inputs += train_raw_seqs[i]
    # if (train_test == "train" and supv_unspv == "supv"):
    valid_inputs, valid_raw_seqs, valid_targets = all_inputs[num_train:], all_raw_seqs[num_train:], targets[num_train:]
    valid_raw_inputs = []
    for i in tqdm(range(len(valid_raw_seqs))):
        valid_raw_inputs += valid_raw_seqs[i]

    test_raw_inputs = []
    for i in tqdm(range(len(test_raw_seqs))):
        test_raw_inputs += test_raw_seqs[i]

    ##Commented as as of now no test file is there for unsupervised learning
    # test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_data(test_file)

    num_words_train_set = numwordstrainset(train_inputs)
    # if (train_test == "train" and supv_unspv == "supv"):
    num_words_valid_set = numwordsvalidset(valid_inputs)

    print(num_words_train_set)
    # print(num_words_valid_set)
    word2id, id2word, label2id, id2label, char2id, id2char = generate_vocab(all_raw_inputs, all_raw_inputs_test, targets)  # , char2id, id2char

    print(char2id)

    train_sentence_sequence = prepare_character_sequence(train_inputs, word2id, char2id)

    train_word_sentence_sequence = prepare_sequence(train_inputs, word2id, char2id)

    # if (train_test == "train" and supv_unspv == "supv"):

    valid_sentence_sequence = prepare_character_sequence(valid_inputs, word2id, char2id)
    valid_word_sentence_sequence = prepare_sequence(valid_inputs, word2id, char2id)

    # Addition for test data
    test_sentence_sequence = prepare_character_sequence(test_inputs, word2id, char2id)

    test_word_sentence_sequence = prepare_sequence(test_inputs, word2id, char2id)
    ###Addition end

    lengths = [len(seq) for seq in train_inputs]
    max_length = max(lengths)

    # if (train_test == "train" and supv_unspv == "supv"):
    lengths_valid = [len(seq) for seq in valid_inputs]
    max_length_valid = max(lengths_valid)

    lengths = [len(seq) for seq in test_inputs]
    max_length_test = max(lengths)

    if len(index_label_dic) != 0:
        index_label_dic = dict(sorted(index_label_dic.items()))

    new_labels = []
    if len(index_label_dic) != 0:
        for i, j in index_label_dic.items():
            new_labels.append(j)

    Overwrite_label = new_labels

    # bigrams = prepare_bigrams(train_inputs,word2id)

    # bigrams_validation = prepare_bigrams_validation(valid_inputs,word2id)

    train_dataset = Dataset(train_inputs, train_raw_seqs, train_sentence_sequence, word2id, id2word, label2id, id2label,
                            train_targets, char2id, id2char, max_length, train_word_sentence_sequence, Overwrite_label,
                            index_label_dic)
    # if (train_test == "train" and supv_unspv == "supv"):
    valid_dataset = Dataset(valid_inputs, valid_raw_seqs, valid_sentence_sequence, word2id, id2word, label2id, id2label,
                            valid_targets, char2id, id2char, max_length_valid, valid_word_sentence_sequence,
                            Overwrite_label, index_label_dic)

    ###Added for test data
    test_dataset = Dataset(test_inputs, test_raw_seqs, test_sentence_sequence, word2id, id2word, label2id, id2label,
                           test_targets, char2id, id2char, max_length_test, test_word_sentence_sequence,
                           Overwrite_label, index_label_dic)
    ###Addidtion ends here

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                               shuffle=True)
    if train_test == "train" and supv_unspv == "supv":
        valid_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                                   shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                              shuffle=False)
    if train_test == "train" and supv_unspv == "supv":
        return train_loader, word2id, id2word, char2id, id2char, train_sentence_sequence, max_length, num_words_train_set, train_inputs, label2id, id2label, num_words_valid_set, valid_loader, valid_sentence_sequence, max_length_valid, valid_word_sentence_sequence
    else:
        return train_loader, word2id, id2word, char2id, id2char, train_sentence_sequence, max_length, num_words_train_set, train_inputs, label2id, id2label, train_word_sentence_sequence, test_loader


def prepare_infer_dataset(train_test, test_file, batch_size, eval_batch_size, Overwrite_label,
                    index_label_dic):

    ###Added for test dataset
    test_inputs, all_raw_inputs_test, test_raw_seqs, test_targets = read_data(test_file)
    ###Addition end

    test_raw_inputs = []
    for i in tqdm(range(len(test_raw_seqs))):
        test_raw_inputs += test_raw_seqs[i]

    word2id, id2word, label2id, id2label, char2id, id2char = generate_vocab(all_raw_inputs, all_raw_inputs_test,
                                                                            targets)  # , char2id, id2char

    # print(char2id)



    # Addition for test data
    test_sentence_sequence = prepare_character_sequence(test_inputs, word2id, char2id)

    test_word_sentence_sequence = prepare_sequence(test_inputs, word2id, char2id)
    ###Addition end

    lengths = [len(seq) for seq in test_inputs]
    max_length_test = max(lengths)

    if len(index_label_dic) != 0:
        index_label_dic = dict(sorted(index_label_dic.items()))

    new_labels = []
    if len(index_label_dic) != 0:
        for i, j in index_label_dic.items():
            new_labels.append(j)

    Overwrite_label = new_labels

    ###Added for test data
    # test_inputs, all_raw_inputs_test, test_raw_seqs, test_targets
    test_dataset = Dataset(test_inputs, test_raw_seqs, test_sentence_sequence, word2id, id2word, label2id, id2label,
                           test_targets, char2id, id2char, max_length_test, test_word_sentence_sequence,
                           Overwrite_label, index_label_dic)
    ###Addidtion ends here

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=collate_fn,
                                              shuffle=False)
    return test_loader


def numwordstrainset(train_raw_inputs):
    words = set()
    for i in range(len(train_raw_inputs)):
        for word in train_raw_inputs[i]:
            if word not in words:
                words.add(word)
    listofword = list(words)
    return len(listofword)


def numwordsvalidset(valid_raw_inputs):
    words = set()
    for i in range(len(valid_raw_inputs)):
        for word in valid_raw_inputs[i]:
            if word not in words:
                words.add(word)
    listofword = list(words)
    return len(listofword)


def generate_vocab(raw_inputs, all_raw_inputs_test, targets):
    word2id, id2word = {}, {}
    char2id, id2char = {}, {}

    label2id, id2label = {}, {}

    # train_inputs, train_raw_inputs, train_raw_seqs = read_data(train_file)
    # valid_inputs, valid_targets, valid_raw_inputs, valid_raw_seqs = read_data(validation_file)
    # test_inputs, test_targets, test_raw_inputs, test_raw_seqs = read_data(test_file)

    # WORD-LEVEL
    word_list = ["<pad>"]  # <emo> is found
    for i in range(len(word_list)):
        word2id[word_list[i]] = len(word2id)
        id2word[len(id2word)] = word_list[i]

    for i in range(len(raw_inputs)):
        # for word in train_inputs[i]:
        if raw_inputs[i] not in word2id:
            word2id[raw_inputs[i]] = len(word2id)
            id2word[len(id2word)] = raw_inputs[i]

    for i in range(len(all_raw_inputs_test)):
        # for word in train_inputs[i]:
        if all_raw_inputs_test[i] not in word2id:
            word2id[all_raw_inputs_test[i]] = len(word2id)
            id2word[len(id2word)] = all_raw_inputs_test[i]

    # for i in range(len(word_list)):
    #     for word in word_list[i]:
    #         for char in word:
    #             if char not in char2id:
    #                 char2id[char] = len(char2id)
    #                 id2char[len(id2char)] = char
    character_list = ["<UNK>", ' ']
    for i in range(len(character_list)):
        char2id[character_list[i]] = len(char2id)
        id2char[len(id2char)] = character_list[i]

    for word in raw_inputs:
        for char in word:
            if char not in char2id:
                char2id[char] = len(char2id)
                id2char[len(id2char)] = char

    for word in all_raw_inputs_test:
        for char in word:
            if char not in char2id:
                char2id[char] = len(char2id)
                id2char[len(id2char)] = char

    # LABEL
    targets = [item for sublist in targets for item in sublist]
    targets = list(set(targets))
    for label in targets:
        if label not in label2id:
            label2id[label] = len(label2id)
            id2label[len(id2label)] = label

    # for i in range(len(valid_targets)):
    #     for word in valid_targets[i]:
    #         if label not in label2id:
    #             label2id[label] = len(label2id)
    #             id2label[len(id2label)] = label

    return word2id, id2word, label2id, id2label, char2id, id2char


def collate_fn(data):
    def merge(sequences, max_length):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.FloatTensor(seq[:end])

        return padded_seqs, lengths

    def merge_sentence(sequences):
        lengths = [len(seq) for seq in sequences]
        final_sequence = []
        # # create an empty matrix with padding tokens
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.FloatTensor(seq[:end])
        lengths.sort(reverse=True)
        return padded_seqs, lengths

    def merge_labels(sequences):
        lengths = [len(seq) for seq in sequences]

        # create an empty matrix with padding tokens
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.FloatTensor(seq[:end])
        lengths.sort(reverse=True)
        return padded_seqs, lengths

    def merge_bigram_word(sequences, max_length):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.FloatTensor(seq[:end])
        return padded_seqs, lengths

    def merge_bigram_label(sequences, max_length):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = torch.FloatTensor(seq[:end])
        return padded_seqs, lengths

    def merge_char(sequences):
        max_char_length = max(max_char_length, len(seq[j]))
        # print(">>>>>>>>", seq[j]weights, max_char_length)
        char_lengths.append(word_arr)
        # print(">max_char_length:", max_char_length)
        padded_seqs = torch.zeros(len(sequences), max(lengths), max_char_length).long()
        for i, seq in enumerate(sequences):
            for j, word in enumerate(sequences[i]):
                wordlen = len(word)
                padded_seqs[i, j, :wordlen] = torch.FloatTensor(sequences[i][j])
        return padded_seqs, char_lengths

    # data.sort(key=lambda x:len(x[0]), reverse=True)
    word_x, raw_x, train_sequence, max_length, index_list, word_target_id, train_word_sentence_sequence_id, overwrite_label = zip(
        *data)  # word_x, char_x, raw_x, train_sequence,
    word_x, x_len = merge(word_x, max_length)
    word_x = torch.LongTensor(word_x)

    word_y, y_len = merge_labels(word_target_id)

    if len(overwrite_label) != 0:
        overwrite_label, overwrite_label_len = merge_labels(overwrite_label)

    word_y = torch.LongTensor(word_y)

    sentence, sentence_lengths = merge_sentence(train_sequence)
    sentence = torch.LongTensor(sentence)

    sentence_word, sentence_word_lengths = merge_sentence(train_word_sentence_sequence_id)

    return word_x, x_len, raw_x, sentence, sentence_lengths, index_list, word_y, y_len, sentence_word, sentence_word_lengths, overwrite_label, overwrite_label_len  # word_y, bpe_ids,


class Dataset(dataset):
    def __init__(self, inputs, raw_inputs, train_sentence_sequence, word2id, id2word, label2id, id2label, targets,
                 char2id, id2char, max_length, train_word_sentence_sequence, Overwrite_label, index_label_dic):
        self.inputs = inputs
        # self.targets = targets
        self.raw_inputs = raw_inputs
        self.train_sentence_sequence = train_sentence_sequence
        self.word2id = word2id
        self.id2word = id2word
        self.max_length = max_length
        self.label2id = label2id
        self.id2label = id2label
        self.targets = targets
        self.train_word_sentence_sequence = train_word_sentence_sequence
        self.Overwrite_label = Overwrite_label

        if len(index_label_dic) != 0:
            self.index_label_dic = index_label_dic

    def __getitem__(self, index):
        index_list = []

        word_input_id, word_target_id = self.vectorize(self.inputs[index], self.targets[index])

        overwrite_label_id = []

        if len(self.Overwrite_label) != 0:
            overwrite_label_id.append(self.Overwrite_label[index])
        raw_input_id = self.raw_inputs[index]
        train_sentence_sequence_id = self.train_sentence_sequence[index]
        train_word_sentence_sequence_id = self.train_word_sentence_sequence[index]
        index_list.append(index)
        return word_input_id, raw_input_id, train_sentence_sequence_id, self.max_length, index_list, word_target_id, train_word_sentence_sequence_id, overwrite_label_id

    def __len__(self):
        return len(self.train_sentence_sequence)


    def vectorize(self, input, target):
        word_input_id = []
        char_input_id = []
        word_target_id = []

        overwrite_label = []
        for i in range(len(input)):
            word_input_id.append(self.word2id[input[i]])
            # char_arr_id = []
            # for char in input[i]:
            #     char_arr_id.append(self.char2id[char])
            # char_input_id.append(char_arr_id)
        for i in range(len(target)):
            word_target_id.append(self.label2id[target[i]])

        return word_input_id, word_target_id  # ,char_input_id


