##################****************##############
###Author - Koustava Goswami################
##################****************##############

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
# from src.model import MetaEmbeddingCLUSTER
from src import constant
import numpy as np
from src.embedder import Embedder, PositionalEncoder
import json


class Lstm_attention(nn.Module):
    def __init__(self, train_test, nb_layers, embedding_dimension, nb_lstm_units, batch_size, vocab, chara_vocab_size,
                 no_of_classes, supv_unsupv, use_cuda):
        super(Lstm_attention, self).__init__()

        self.iterations = 0
        self.nb_layers = nb_layers
        self.embedding_dimension = embedding_dimension  # 6
        self.nb_lstm_units = nb_lstm_units
        self.batch_size = batch_size
        self.vocab = vocab

        self.no_of_classes = no_of_classes

        if supv_unsupv == "unsupv":
            self.clusterCenter = nn.Parameter(torch.zeros(4, 4))

        self.alpha = 1.0

        nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab['<pad>']

        self.char_embeddings = nn.Embedding(chara_vocab_size, self.embedding_dimension)

        if train_test == "train":
            self.char_embeddings.weight.requires_grad = True
            torch.nn.init.xavier_uniform_(self.char_embeddings.weight)

        self.conv1 = nn.Sequential(nn.Conv1d(self.embedding_dimension, 128, kernel_size=2, padding=0), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv1d(self.embedding_dimension, 128, kernel_size=3, padding=0), nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv1d(self.embedding_dimension, 128, kernel_size=4, padding=0), nn.ReLU())

        self.conv4 = nn.Sequential(nn.Conv1d(self.embedding_dimension, 128, kernel_size=5, padding=0), nn.ReLU())

        self.conv5 = nn.Sequential(nn.Conv1d(self.embedding_dimension, 128, kernel_size=6, padding=0), nn.ReLU())

        self.atten1 = Attention(128, batch_first=True)

        self.dense1_bn = nn.BatchNorm1d(128 * 5)

        self.drop = nn.Dropout(p=0.5)

        self.hidden_to_LID = nn.Linear(128 * 5, 1024)

        self.output = nn.Linear(1024, self.no_of_classes)

        if train_test == "train":
            self._create_weights()
        self.use_cuda = use_cuda

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            # if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear) or isinstance(module, nn.LSTM):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                # module.weight.data.normal_(mean, std)
                torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, X, X_lengths, supv_unsupv, x_word, sentence_word_lengths, train_test):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        if self.use_cuda:
            X = X.to('cuda')
            # X_lengths = X_lengths.to('cuda')
            x_word = x_word.to('cuda')

        lst = []
        lts1 = []
        final_sentence_tensor = []
        batch_size, seq_len = X.size()

        word_batch_size, word_seq_len = x_word.size()
        X = self.char_embeddings(X)
        X = X.permute(0, 2, 1)
        c1 = self.conv1(X)
        c1 = c1.permute(0, 2, 1)
        X_C1, attention = self.atten1(c1)
        c2 = self.conv2(X)
        c2 = c2.permute(0, 2, 1)
        X_C2, attention = self.atten1(c2)
        c3 = self.conv3(X)
        c3 = c3.permute(0, 2, 1)
        X_C3, attention = self.atten1(c3)
        c4 = self.conv4(X)
        c4 = c4.permute(0, 2, 1)
        X_C4, attention = self.atten1(c4)
        print('c4', c4.shape)
        c5 = self.conv5(X)
        c5 = c5.permute(0, 2, 1)
        print('C5', c5.shape)
        X_C5, attention = self.atten1(c5)

        if (X_C1.size()[0] == 128 or X_C2.size()[0] == 128 or X_C3.size()[0] == 128 or X_C4.size()[0] == 128 or
                X_C5.size()[0] == 128):
            X_C1 = X_C1.unsqueeze(0)
            X_C2 = X_C2.unsqueeze(0)
            X_C3 = X_C3.unsqueeze(0)
            X_C4 = X_C4.unsqueeze(0)
            X_C5 = X_C5.unsqueeze(0)

        concatenated = torch.cat((X_C1, X_C2, X_C3, X_C4, X_C5), 1)
        concatenated = self.dense1_bn(concatenated)
        concatenated = self.drop(concatenated)
        X = torch.tanh(self.hidden_to_LID(concatenated))

        if supv_unsupv == "supv":
            x = self.output(X)
            X_out = X
        else:
            X = self.output(X)
            X_out = X
            x = F.softmax(X, dim=1)

        Y_hat = x
        return Y_hat, X_out  # ,X_word_label

    def updateClusterCenter(self, cc):
        """
        To update the cluster center. This is a method for pre-train phase.
        When a center is being provided by kmeans, we need to update it so
        that it is available for further training
        :param cc: the cluster centers to update, size of num_classes x num_features
        """
        self.clusterCenter.data = torch.from_numpy(cc)

    def getTDistribution(self, x, clusterCenter):
        """
        student t-distribution, as same as used in t-SNE algorithm.
         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
         
         :param x: input data, in this context it is encoder output
         :param clusterCenter: the cluster center from kmeans
         """
        if self.use_cuda:
            xe = torch.unsqueeze(x, 1).cuda() - clusterCenter.cuda()
        else:
            xe = torch.unsqueeze(x, 1).to(torch.device("cpu")) - clusterCenter.to(torch.device("cpu"))

        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe, xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t()  # due to divison, we need to transpose q
        return q

    def getDistance(self, x, clusterCenter, alpha=1.0):
        """
        it should minimize the distince to 
         """
        if not hasattr(self, 'clusterCenter'):
            self.clusterCenter = nn.Parameter(torch.zeros(4, 4))  ## change the number of class and centers
        if self.use_cuda:
            xe = torch.unsqueeze(x, 1).cuda() - clusterCenter.cuda()
        else:
            xe = torch.unsqueeze(x, 1).to(torch.device("cpu")) - clusterCenter.to(torch.device("cpu"))
        # need to sum up all the point to the same center - axis 1
        d = torch.sum(torch.mul(xe, xe), 2)
        return d


# attention layer code inspired from: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
class Attention(nn.Module):
    def __init__(self, hidden_size, batch_first=False):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first

        modules = []

        # last attention layer must output 1
        modules.append(nn.Linear(hidden_size, 1))
        modules.append(nn.Tanh())

        self.attention = nn.Sequential(*modules)

        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self):
        pass

    def forward(self, inputs):  # , lengths
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        scores = self.attention(inputs).squeeze()
        scores = self.softmax(scores)

        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, scores
