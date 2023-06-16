
import json                                                                                                               
import numpy as np                                                                      
from collections import Counter                                                         
from torch.utils import data
# from torchtext import data
import torch                                                                           
#from torch.utils.data import Dataset                                                  
import pandas as pd                                                                     
from tqdm import tqdm                                                                    
                                                                                        
from keras.preprocessing.text import Tokenizer                                          
from keras.preprocessing.sequence import pad_sequences                                  
from nltk.tokenize import word_tokenize 

from nltk.cluster import KMeansClusterer                                                
import nltk                                                                             
import numpy as np                                                                     
                                                                                        
from sklearn import cluster                                                            
from sklearn import metrics                                                             
                            

torch.cuda.set_device(1) #Change the cuda number

def read_data(file_path, stemming_arabic=False, new_preprocess=False):                  
    inputs, targets, raw_inputs, raw_seqs = [], [], [], []                              
    with open(file_path, "r", encoding="utf-8") as file_in:                             
        input_seq, target_seq, raw_seq = [], [], []                                     
        for line in file_in:                                                            
            word_vocab=(word_tokenize(line))                                            
            for w in range(len(word_vocab)):                                            
                input_seq.append(word_vocab[w])                                         
                raw_seq.append(word_vocab[w])                                           
                raw_inputs.append(word_vocab[w])                                        
            if len(input_seq) > 0:                                                      
                inputs.append(input_seq)                                                
                raw_seqs.append(raw_seq)                                                
                input_seq, target_seq, raw_seq =  [], [], []                            
    return inputs,raw_inputs, raw_seqs



def prepare_sentence(train_file):
    inputs,raw_inputs, raw_seqs =  read_data(train_file)

    return inputs,raw_inputs, raw_seqs

    #print(raw_seqs)


inputs,raw_inputs, raw_seqs = prepare_sentence('/home/kougos/data/language_detection_dataset/fasttext_k-means/text_without_labels.txt')

#Here I have made a dictionary to get the embeddings from the glove vector reading the 50 dimensional data
embeddings_index = dict()
f = open('/home/kougos/data/language_detection_dataset/fasttext_k-means/embedding_file.vec', 'r', encoding='utf8', errors='ignore')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

mylist=[]
semb =[]
for i in range(len(raw_seqs)):
    print('Starting ',i,' th sentence')
    for j in raw_seqs[i]:
        if(j in embeddings_index.keys()):
            x=embeddings_index[j]
            x = torch.FloatTensor(x)
            x= x.cuda()
            mylist.append(x)
    sentence_embedding = torch.mean(torch.stack(mylist), dim=0).cuda()
    sentence_embedding = sentence_embedding.cpu().detach().numpy()
    #nx, ny = sentence_embedding.shape
    #sentence_embedding = sentence_embedding.reshape(nx*ny)
    semb.append(sentence_embedding)

print('Preparing for clustering') 

NUM_CLUSTERS=11                                                                         
kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=300,avoid_empty_clusters=True)                                                    
assigned_clusters = kclusterer.cluster(semb, assign_clusters=True)                      
print (assigned_clusters)                                                               
                                                                                        
#a =[]                                                                                 
#for i in range(len(raw_seqs)):                                                    
    #if(i<=4):                                                                       
        #a.append(raw_seqs[i])                                                     
                                                                                        
dictionary_sentence ={}                                                                 
for index, sentence in enumerate(raw_seqs):                                         
    dictionary_sentence.update({str(sentence) : str(assigned_clusters[index])})         
    #print (str(assigned_clusters[index]) + ":" + str(sentence))                        
                                                                                        
print('Copying to file')                                                                
                                                                                        
with open('file.txt', 'w') as file:                                                     
     file.write(json.dumps(dictionary_sentence, indent=2))           


kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(semb)
 
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


print ("Cluster id labels for inputted data")
print (labels)
print ("Centroids data")
print (centroids)
 
print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
print (kmeans.score(semb))
 
silhouette_score = metrics.silhouette_score(semb, labels, metric='euclidean')
 
print ("Silhouette_score: ")
print (silhouette_score)



dictionary_sentence_sklearn ={}                                                                 
for index, sentence in enumerate(raw_seqs):                                         
    dictionary_sentence_sklearn.update({str(sentence) : str(labels[index])})         
    #print (str(assigned_clusters[index]) + ":" + str(sentence))   

print('Copying to file')                                                                
                                                                                        
with open('file_sklearn.txt', 'w') as file:                                                     
     file.write(json.dumps(dictionary_sentence_sklearn, indent=2)) 



