from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.optim as optim
import sys

import logging

from tqdm import tqdm
import numpy as np

# from src.Vocab import prepare_dataset
from src.Vocab_char import prepare_dataset
from src import constant
from trainers.trainer import Trainer
# from src.ngrammodel import NgramModel
# from src.Lstm_attention import Lstm_attention
# from src.charac_word_lstm_attention import Lstm_attention
from src.char_attention import Lstm_attention
# from src.model import MetaEmbeddingCLUSTER
from src.train_common import compute_num_params, lr_decay_map
# from src.Autoencoder import DEC_AE

from nltk.cluster import KMeansClusterer
import nltk
import numpy as np

from sklearn import cluster
from sklearn import metrics

import json

print("###### Initialize trainer  and to do that uncomment ######")
trainer = Trainer()

eval_batch_size = 10
# train_loader, word2id, id2word,train_sentence_sequence,max_length,num_words_train_set,train_inputs,weights_matrix,label2id, id2label,num_words_valid_set,valid_loader,valid_sentence_sequence,max_length_valid = prepare_dataset(constant.params["train_file"],constant.params["batch_size"],eval_batch_size) #num_words_valid_set,valid_loader,valid_sentence_sequence,max_length_valid,
# if(constant.params["train_test"] == "train" and constant.params["supv_upnspv"] == "supv"):
#     train_loader, word2id, id2word,char2id, id2char,train_sentence_sequence,max_length,num_words_train_set,train_inputs,label2id, id2label,num_words_valid_set,valid_loader,valid_sentence_sequence,max_length_valid,valid_word_sentence_sequence = prepare_dataset(constant.params["supv_upnspv"],constant.params["train_test"],constant.params["train_file"],constant.params["test_file"],constant.params["batch_size"],eval_batch_size,Overwrite_label=[],index_label_dic={})
# else:
train_loader, word2id, id2word, char2id, id2char, train_sentence_sequence, max_length, num_words_train_set, train_inputs, label2id, id2label, train_word_sentence_sequence, test_loader = prepare_dataset(
    constant.params["supv_upnspv"], constant.params["train_test"], constant.params["train_file"],
    constant.params["test_file"], constant.params["batch_size"], eval_batch_size, Overwrite_label=[],
    index_label_dic={})

print("###### Prepare the dataset ######")

emb_size = constant.params["embedding_size_word"]
char_emb_size = constant.params["embedding_size_char"]
hidden_size = constant.params["hidden_size"]
lstm_units = constant.params["lstm_units"]
lstm_layers = constant.params["lstm_layers"]
no_of_classes = constant.params["no_of_classes"]
char_hidden_size = constant.params["embedding_size_char_per_word"]
num_layers = constant.params["num_layers"]
num_heads = constant.params["num_heads"]
dim_key = constant.params["dim_key"]
dim_value = constant.params["dim_value"]
filter_size = constant.params["filter_size"]
max_length = constant.params["max_length"]
input_dropout = constant.params["input_dropout"]
dropout = constant.params["dropout"]
attn_dropout = constant.params["attn_dropout"]
relu_dropout = constant.params["relu_dropout"]
add_emb = constant.params["add_emb"]
no_word_emb = constant.params["no_word_emb"]
add_char_emb = constant.params["add_char_emb"]

emb_list = constant.params["emb_list"]

train_test = constant.params["train_test"]

supv_unsupv = constant.params["supv_upnspv"]

bpe_emb_size = constant.params["bpe_emb_size"]
bpe_hidden_size = constant.params["bpe_hidden_size"]
bpe_lang_list = [] if constant.params["bpe_lang_list"] is None else constant.params["bpe_lang_list"]
bpe_emb_size = constant.params["bpe_emb_size"]
bpe_vocab = constant.params["bpe_vocab"]
mode = constant.params["mode"]
no_projection = constant.params["no_projection"]
use_crf = constant.params["use_crf"]

if constant.USE_CUDA:
    torch.cuda.set_device(1)  # Change the cuda number
else:
    device = torch.device("cpu")
pad_idx = 0

context_size = 2  # As we are working with bigram

print("######    Start training   ######")

# model = Lstm_attention(train_test = train_test,nb_layers=lstm_layers,embedding_dimension=emb_size,nb_lstm_units=lstm_units,batch_size=constant.params["batch_size"],vocab=word2id,weight_matrix=weights_matrix)
# model = Lstm_attention(train_test = train_test,nb_layers=lstm_layers,embedding_dimension=emb_size,nb_lstm_units=lstm_units,batch_size=constant.params["batch_size"],vocab=word2id,chara_vocab_size = len(char2id),no_of_classes=no_of_classes,supv_unsupv=supv_unsupv)

PATH = "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"])
# model.load_state_dict(torch.load(PATH))
model = torch.load(PATH)

if constant.USE_CUDA:
    model = model.cuda()

for param in model.state_dict():
    print(param)

model.eval()

for param in model.state_dict():
    print(param)

for parameter in model.parameters():
    parameter.requires_grad = False

# if constant.params["supv_upnspv"] == "supv":
#     _,accuracy = trainer.predict_supv(model,train_loader,word2id,id2word,train_sentence_sequence,num_words_train_set,len(train_sentence_sequence),supv_unsupv,test_loader)
# else:
y_pred, index_dict = trainer.predict(model, train_loader, word2id, id2word, train_sentence_sequence,
                                     num_words_train_set, len(train_sentence_sequence), supv_unsupv, test_loader,
                                     train_test)
print(f"\n{y_pred=}\n\n{index_dict=}")

# index_dict = dict(sorted(index_dict.items()))
#
# import operator
# data_dictionary={}
# item_list=[]
# for k,v in index_dict.items():
#     item_list.append(v)
#     data_dictionary.update({str(train_inputs[k]): str(v)})
#
# print(set(item_list))
#
# print('Copying to file')
#
# with open('file_unspv1_german.txt', 'w',encoding='utf8') as file:
#     json.dump(data_dictionary, file,ensure_ascii=False,indent=2)
# print('File writing done')
