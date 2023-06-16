import pandas as pd 
import ast

import json                                                                                                               
import numpy as np                                                                      
from collections import Counter                                                         
#from torch.utils.data import Dataset                                                   
from torchtext import data                                                             
import torch                                                                           
#from torch.utils.data import Dataset                                                  
import pandas as pd                                                                     
from tqdm import tqdm                                                                    
                                                                                                                         
from nltk.tokenize import word_tokenize 

from nltk.cluster import KMeansClusterer                                                
import nltk                                                                             
import numpy as np                                                                     
                                                                                        
from sklearn import cluster                                                            
from sklearn import metrics

import collections

f = open('/home/kougos/data/language_detection_dataset/fasttext_k-means/text_without_labels.txt', "r") 
df = pd.read_csv('train_full_3k.csv', sep='\t',header=None)
df = df.drop_duplicates()
#df = df.iloc[1:]
#label = df[0].values.tolist()
#text=df[1].values.tolist()
#a = len(label)
#print(a)
#print(label[:1])



def read_data(file_path, stemming_arabic=False, new_preprocess=False):                  
    inputs, targets, raw_inputs, raw_seqs,label = [], [], [], [] , []                             
    with open(file_path, "r", encoding="utf-8") as file_in:                             
        input_seq, target_seq, raw_seq = [], [], [] 
        #file_in = file_in[0:25000]                                    
        for line in file_in:   
            line = line.replace('\n', '')
            #arr = line.split(',') 
            arr = line.split('\t')
            lanid, sent = arr[0], arr[1] 
            if(sent != " text"):                                                       
                word_vocab=(word_tokenize(sent))                                            
                for w in range(len(word_vocab)):
                    if(('``')!=  word_vocab[w] and ("''")!= word_vocab[w]):                                           
                        input_seq.append(word_vocab[w])                                         
                        raw_seq.append(word_vocab[w])                                           
                        raw_inputs.append(word_vocab[w]) 
                target_seq.append(lanid)                                      
            if len(input_seq) > 0:                                                      
                
                if(input_seq not in inputs):
                    inputs.append(input_seq)                                                 
                if(raw_seq not in raw_seqs):
                    raw_seqs.append(raw_seq)  
                    targets.append(target_seq) 
                    label.append(lanid)                                              
                input_seq, target_seq, raw_seq =  [], [], []                            
    return inputs,raw_inputs, raw_seqs,targets,label



def prepare_sentence(train_file):
    inputs,raw_inputs, raw_seqs,targets,label =  read_data(train_file)

    return inputs,raw_inputs, raw_seqs,targets,label

    #print(raw_seqs)


def purity():

    with open('/home/kougos/data/language_detection_dataset/supervised_code_model/fasttext_k-means/file_unspv1_german.txt', 'r') as f:
        s = f.read()
        whip = ast.literal_eval(s)

    print(len(whip.keys()))






    inputs,raw_inputs, raw_seqs,targets,label = prepare_sentence('/home/kougos/data/language_detection_dataset/supervised_code_model/fasttext_k-means/german_dataset.csv')

    print(len(raw_seqs))
    len(label)

    a = len(raw_seqs)
    sen=[]
    for i in whip.keys():
        sen.append(i)

    clusters=[]
    for k,v in whip.items():
        clusters.append(ast.literal_eval(v))
    print (set(clusters))

    col_names =  ['Text', 'Lang_LB', 'Clusters']
    my_df  = pd.DataFrame(columns = col_names)
    my_df

    unattended_sentences =[]
    for i in range(len(raw_seqs)):
        j = ast.literal_eval(sen[i])
        if(raw_seqs[i]==j):
            clsus = clusters[i]
            lbl = label[i]
            my_df.loc[len(my_df)] = [j, lbl, clsus]
        else:
            unattended_sentences.append(raw_seqs[i])

    x = my_df.groupby('Clusters').agg(lambda x: x.tolist()).reset_index()


    free=[]
    for i in range(len(x['Clusters'])):
        try:
            counter=collections.Counter(x['Lang_LB'][i])
            dicti=dict(counter)
            print(x['Clusters'][i], ' : ',dicti)
            free.append(max(dicti.values()))
        except:
            print('Not all the clusters are there Only ', len(set(clusters)),' fund where there are total ', len(label), ' number labels there')
        
    purity = sum(free)/len(raw_seqs)

    print("The purity is ",purity)

    return purity

#purity()