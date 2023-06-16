##################****************##############
###Author - Koustava Goswami################
##################****************##############

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import logging
import numpy as np
from tqdm import tqdm
import math
import statistics

from src.ngrammodel import NgramModel
from src import constant
from src.train_common import compute_num_params, lr_decay_map
import operator
import itertools
# from cluster_purity import purity
from src.Vocab_char import prepare_dataset

import operator

from torch.autograd import Variable

import json

from nltk.cluster import KMeansClusterer
import nltk
import numpy as np

from sklearn import cluster
from sklearn import metrics
from logger import Logger

from sklearn.cluster import MiniBatchKMeans, KMeans

from torchviz import make_dot

from src.embedder import Embedder

import torch.nn.functional as F

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.metrics import confusion_matrix

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

logger = Logger('./logs')
if not constant.USE_CUDA:
    device = torch.device("cpu")

class cust1(torch.nn.Module):

    def __init__(self):
        super(cust1, self).__init__()

    def forward(self, prob_dist):  # ,X_word_label
        xdict = []
        sums = 0
        total_sum = []
        squr_mean = []
        var_sqaure = []
        sumvariance = 0
        sum_squr_mean = 0
        mean_squ = 0
        word_label = []
        xdict1 = []

        # changing 30*11 matrix to 11*30 matrix
        prob_dist = prob_dist.permute(1, 0)

        _, tag_size = prob_dist.size()

        loss = torch.min(torch.abs(((1 / (4 - 1)) * torch.sum(torch.sum(prob_dist, dim=0).pow(2))) - (
                    (4 / (4 - 1)) * (1 / 4 * torch.sum(prob_dist, dim=0)).pow(2))))

        return loss
        # torch.mean(torch.stack(xdict))#min([ p for p in xdict])#max([ p for p in xdict]) #torch.mean(torch.stack(xdict))


class cust2(torch.nn.Module):

    def __init__(self):
        super(cust2, self).__init__()

    def forward(self, prob_dist):  # ,X_word_label
        xdict = []
        sums = 0
        total_sum = []
        squr_mean = []
        var_sqaure = []
        sumvariance = 0
        sum_squr_mean = 0
        mean_squ = 0
        word_label = []
        xdict1 = []

        batch_size, tag_size = prob_dist.size()

        loss = torch.sum(torch.max(prob_dist, 1)[0]) - (torch.max(torch.sum(prob_dist.pow(2), dim=0)))

        return loss


class Trainer():

    def __init__(self):
        super(Trainer, self).__init__()

        self.clusterCenter = nn.Parameter(torch.zeros(3, 3))
        self.alpha = 1.0

    @staticmethod
    def target_distribution(q):
        weight = (q ** 2) / q.sum(0)
        # print('q',q)
        return Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)

    def logAccuracy(self, pred, label):
        print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
              % (acc(label, pred), nmi(label, pred)))

    @staticmethod
    def kld(q, p):
        return torch.sum(p * torch.log(p / q), dim=-1)

    @staticmethod
    def cross_entropy(q, p):
        return torch.sum(torch.sum(p * torch.log(1 / (q + 1e-7)), dim=-1))

    @staticmethod
    def depict_q(p):
        q1 = p / torch.sqrt(torch.sum(p, dim=0))
        qik = q1 / q1.sum()
        return qik

    @staticmethod
    def distincePerClusterCenter(dist):
        totalDist = torch.sum(torch.sum(dist, dim=0) / (torch.max(dist) * dist.size(1)))
        return totalDist

    def clustering(self, mbk, x, model, sentence_lengths, supv_unsupv, sentence_word, sentence_word_lengths):
        model.eval()
        y_pred_ae, X_Out = model(x, sentence_lengths, supv_unsupv, sentence_word, sentence_word_lengths)
        y_pred_ae = X_Out.data.cpu().numpy()
        y_pred = mbk.partial_fit(y_pred_ae)  # seems we can only get a centre from batch
        self.cluster_centers = mbk.cluster_centers_  # keep the cluster centers
        model.updateClusterCenter(self.cluster_centers)

    def train(self, model, train_loader, word2id, id2word, train_sentence_sequence, num_words_train_set,
              num_words_valid_set, valid_loader, valid_sentence_sequence, max_length_valid, supv_unsupv, train_inputs,
              train_test, train_file, batch_size, eval_batch_size):
        """
        Train a model on a dataset
        """

        # print("<pad>",word2id["<pad>"])

        print("len word vocab:", len(word2id))

        if constant.USE_CUDA:
            model = model.cuda()
        else:
            model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=constant.params["lr"], weight_decay=0.01, momentum=0.9)

        print("Parameters: {}(trainable), {}(non-trainable)".format(
            compute_num_params(model)[0], compute_num_params(model)[1]))
        # print("Training scheme: {} {}".format(lrd_scheme, lrd_range))
        if constant.USE_CUDA:
            criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() #
        else:
            criterion = nn.CrossEntropyLoss().to(device)  # nn.NLLLoss() #
        iterations, epochs, best_valid_f1, best_valid_loss, best_epoch = 0, 0, 0, 100, 0
        cnt = 0
        start_iter = 0
        global_step = 0
        for epoch in range(constant.params["num_epochs"]):  # constant.params["num_epochs"]
            model.train()
            sys.stdout.flush()
            log_loss = []
            running_loss = 0
            correct = 0
            total_train = 0
            correct_train = 0
            all_embs = []
            dic = {}
            count = 0
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            max_values = []
            for i, (data) in enumerate(pbar, start=start_iter):
                word_x, x_len, raw_x, sentence, sentence_lengths, index_list, word_y, y_len, sentence_word, sentence_word_lengths, overwrite_label, overwrite_label_len = data  # word_x, char_x, x_len, raw_x, sentence,
                # print(f"{word_x=}")
                if constant.USE_CUDA:
                    word_x = word_x.cuda()
                    sentence = sentence.cuda()
                    sentence_word = sentence_word.cuda()
                    word_y = word_y.cuda()
                    word_y = word_y.squeeze()
                    optimizer.zero_grad()
                    output, XX = model(sentence, sentence_lengths, supv_unsupv, sentence_word, sentence_word_lengths,
                                       train_test)  # ,word_x ,X_word_label

                    loss = criterion(output, word_y)

                    running_loss += loss.item()
                    log_loss.append(loss.item())
                    loss.backward()

                    optimizer.step()

                    _, predicted = torch.max(output.data, 1)
                    total_train += word_y.nelement()
                    correct_train += (predicted == word_y).sum().item()  # predicted.eq(word_y.data).sum().item()
                else:
                    word_x = word_x.to(device)
                    sentence = sentence.to(device)
                    sentence_word = sentence_word.to(device)
                    word_y = word_y.to(device)
                    # print(f"{word_y=}")

                    word_y = word_y.squeeze()

                    optimizer.zero_grad()
                    # print("---------------------------------------------------------------------")
                    # print(f"{sentence=}\n\n{sentence_lengths=}\n\n{sentence_word=}\n\n{sentence_word_lengths=}")
                    # print("---------------------------------------------------------------------")
                    output, XX = model(sentence, sentence_lengths, supv_unsupv, sentence_word, sentence_word_lengths,
                                       train_test)  # ,word_x ,X_word_label
                    # print(f"{output=}")
                    loss = criterion(output, word_y)

                    running_loss += loss.item()
                    log_loss.append(loss.item())
                    loss.backward()

                    optimizer.step()

                    _p, predicted = torch.max(output.data, 1)
                    # print(f"{_p=}")
                    # print(f"{predicted=}")
                    # print(f"{sentence=}")
                    total_train += word_y.nelement()
                    correct_train += (predicted == word_y).sum().item()  # predicted.eq(word_y.data).sum().item()

            # if lr_scheduler_epoch:
            #     lr_scheduler_epoch.step()

            if total_train > 0:
                accuracy = 100 * correct_train / total_train
            else:
                accuracy = 0
            print(set(max_values))

            running_loss_validation, accuracy_validation = self.evaluate(model, train_loader, word2id, id2word,
                                                                         train_sentence_sequence, num_words_train_set,
                                                                         num_words_valid_set, valid_loader,
                                                                         valid_sentence_sequence, max_length_valid,
                                                                         supv_unsupv, train_test)
            print('Loss: {:.6f}'.format(np.mean(log_loss)),
                  "Epoch: {}/{}.. ".format(epoch + 1, constant.params["num_epochs"]), "Accuracy = {}".format(accuracy),
                  'Loss of validation: {:.6f}'.format(np.mean(running_loss_validation)),
                  "Accuracy = {}".format(accuracy_validation))  #
            torch.save(model, "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"]))

        return running_loss / num_words_train_set

    def train_unsupervised(self, model, train_loader, word2id, id2word, train_sentence_sequence, num_words_train_set,
                           num_words_valid_set, supv_unsupv, train_inputs, train_test, train_file, batch_size,
                           eval_batch_size, test_loader):
        """
        Train a model on a dataset
        """

        # print("<pad>",word2id["<pad>"])

        lmbd = 0.9

        print("len word vocab:", len(word2id))

        if constant.USE_CUDA:
            model = model.cuda()
        else:
            model = model.to(device)
        # model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.SGD(model.parameters(), lr=constant.params["lr"], weight_decay=0.01, momentum=0.9)
        # print(list(model.parameters()))
        optimizer = optim.Adam(
            model.parameters(),
            betas=(constant.params["optimizer_adam_beta1"],
                   constant.params["optimizer_adam_beta2"]),
            lr=constant.params["lr"])

        lr_scheduler_step, lr_scheduler_epoch = None, None  # lr schedulers
        lrd_scheme, lrd_range = constant.params["lr_decay"].split('_')
        lrd_func = lr_decay_map()[lrd_scheme]

        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lrd_func(constant.params),
            last_epoch=int(model.iterations) or -1
        )

        lr_scheduler_epoch, lr_scheduler_step = None, None
        if lrd_range == 'epoch':
            lr_scheduler_epoch = lr_scheduler
        elif lrd_range == 'step':
            lr_scheduler_step = lr_scheduler
        else:
            raise ValueError("Unknown lr decay range {}".format(lrd_range))

        print("Parameters: {}(trainable), {}(non-trainable)".format(
            compute_num_params(model)[0], compute_num_params(model)[1]))

        # criterion = cust1()
        criterion2 = cust2()
        if constant.USE_CUDA:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss().to(device)


        loss = self.Feature_extractor(model, train_loader, word2id, id2word, train_sentence_sequence,
                                      num_words_train_set, len(train_sentence_sequence), supv_unsupv, test_loader,
                                      train_test)

        # train_loader, word2id, id2word,char2id, id2char,train_sentence_sequence,max_length,num_words_train_set,train_inputs,label2id, id2label,train_word_sentence_sequence,test_loader = prepare_dataset(supv_unsupv,train_test,train_file,batch_size,eval_batch_size,y_pred,index_dict)

        mbk = MiniBatchKMeans(n_clusters=3, n_init=20, batch_size=batch_size)
        got_cluster_center = False

        iterations, epochs, best_valid_f1, best_valid_loss, best_epoch = 0, 0, 0, 100, 0
        cnt = 0
        start_iter = 0
        model = torch.load("bestModel")
        for epoch in range(constant.params["num_epochs"]):  # constant.params["num_epochs"]

            sys.stdout.flush()
            log_loss = []
            running_loss = 0
            correct = 0
            total_train = 0
            correct_train = 0
            all_embs = []
            dic = {}
            loss = 0
            count = 0

            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            max_values = []

            for i, (data) in enumerate(pbar, start=start_iter):
                word_x, x_len, raw_x, sentence, sentence_lengths, index_list, word_y, y_len, sentence_word, sentence_word_lengths, _, _ = data  # word_x, char_x, x_len, raw_x, sentence,
                max_target = []
                if constant.USE_CUDA:
                    sentence = sentence.cuda()
                    sentence_word = sentence_word.cuda()
                else:
                    sentence = sentence.to(device)
                    sentence_word = sentence_word.to(device)

                optimizer.zero_grad()
                if not got_cluster_center:
                    self.clustering(mbk, sentence, model, sentence_lengths, supv_unsupv, sentence_word,
                                    sentence_word_lengths, )
                    if epoch > 1:
                        got_cluster_center = True
                else:
                    model.train()
                    output, X_out = model(sentence, sentence_lengths, supv_unsupv, sentence_word,
                                          sentence_word_lengths, train_test)
                    q, dist, clssfied = model.getTDistribution(X_out, self.clusterCenter), \
                                        model.getDistance(X_out, model.clusterCenter), \
                                        F.softmax(X_out, dim=1)
                    d = self.distincePerClusterCenter(dist)
                    qik = self.depict_q(clssfied)
                    loss1 = self.cross_entropy(clssfied, qik)
                    loss2 = criterion2(output)
                    loss = loss1

                    running_loss += loss.item()
                    log_loss.append(loss.item())
                    a = list(model.parameters())[0].clone()
                    list(model.parameters())[0].grad

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                    optimizer.step()

                    if lr_scheduler_step:
                        lr_scheduler_step.step()

                    b = list(model.parameters())[0].clone()

                    if torch.equal(a.data, b.data):
                        count = count + 1

            if lr_scheduler_epoch:
                lr_scheduler_epoch.step()
            print(set(max_values))
            print('The number of parameters did not update is : ', count)
            print('Loss: {:.6f}'.format(np.mean(log_loss)),
                  "Epoch: {}/{}.. ".format(epoch + 1, constant.params["num_epochs"]))  #
            torch.save(model.state_dict(),
                       "{}/{}.pt".format(constant.params["model_dir"], constant.params["save_path"]))

            _, _, index_dict = self.predict(model, train_loader, word2id, id2word, train_sentence_sequence,
                                            num_words_train_set, len(train_sentence_sequence), supv_unsupv, test_loader,
                                            train_test)

        return np.mean(log_loss)  # running_loss / num_words_valid_set

    def evaluate(self, model, train_loader, word2id, id2word, train_sentence_sequence, num_words_train_set,
                 num_words_valid_set, valid_loader, valid_sentence_sequence, max_length_valid, supv_unsupv, train_test):

        model.eval()

        start_iter = 0

        index_dict = {}
        running_loss = 0
        maxx = []
        index_length = []
        log_loss = []
        running_loss = 0
        correct = 0
        total_train = 0
        correct_train = 0
        all_embs = []
        dic = {}
        if constant.USE_CUDA:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
        with torch.no_grad():
            for i, (data) in enumerate(pbar, start=start_iter):
                word_x, x_len, raw_x, sentence, sentence_lengths, index_list, word_y, y_len, sentence_word, sentence_word_lengths, overwrite_label, overwrite_label_len = data  # word_x, char_x, x_len, raw_x, sentence,
                if constant.USE_CUDA:
                    word_x = word_x.cuda()
                    sentence = sentence.cuda()
                    sentence_word = sentence_word.cuda()
                    word_y = word_y.cuda()
                else:
                    word_x = word_x.to(device)
                    sentence = sentence.to(device)
                    sentence_word = sentence_word.to(device)
                    word_y = word_y.to(device)

                if word_y.size()[0] == 1:
                    word_y = word_y.squeeze(1)
                else:
                    word_y = word_y.squeeze()
                output, XX = model(sentence, sentence_lengths, supv_unsupv, sentence_word, sentence_word_lengths,
                                   train_test)  # ,word_x ,X_word_label

                loss = criterion(output, word_y)
                running_loss += loss.item()
                log_loss.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total_train += word_y.nelement()
                correct_train += predicted.eq(word_y.data).sum().item()
            if total_train > 0:
                accuracy = 100 * correct_train / total_train
            else:
                accuracy = 0

            print(accuracy)

        return log_loss, accuracy

    def predict(self, model, train_loader, word2id, id2word, train_sentence_sequence, num_words_train_set,
                num_words_valid_set, supv_unsupv, test_loader, train_test):

        model.eval()

        start_iter = 0

        lmbd = 0.9

        index_dict = {}
        running_loss = 0
        maxx = []
        log_loss = []
        index_length = []

        to_eval = []
        true_labels = []
        final_index_list = []
        pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        with torch.no_grad():
            for i, (data) in enumerate(pbar, start=start_iter):
                word_x, x_len, raw_x, sentence, sentence_lengths, index_list, word_y, y_len, sentence_word, sentence_word_lengths, overwrite_label, overwrite_label_len = data  # word_x, char_x, x_len, raw_x, sentence,

                if constant.USE_CUDA:
                    sentence = sentence.cuda()
                    sentence_word = sentence_word.cuda()
                    overwrite_label = overwrite_label.cuda()
                else:
                    sentence = sentence.to(device)
                    sentence_word = sentence_word.to(device)
                    overwrite_label = overwrite_label.to(device)

                output, X_out = model.forward(sentence, sentence_lengths, supv_unsupv, sentence_word,
                                              sentence_word_lengths, train_test)  # ,word_x #,X_word_label

                _, s = word_y.size()

                if s == 2:
                    print("Mistmatch")

                to_eval.append(X_out)
                true_labels.append(word_y)

                ab = itertools.chain(index_list)
                index_list = list(ab)
                index_list = [item for sublist in index_list for item in sublist]
                final_index_list.append(index_list)

            to_eval = torch.cat(to_eval, dim=0)
            true_labels = torch.cat(true_labels, dim=0)
            true_labels = true_labels.squeeze()
            to_eval = to_eval.data.cpu().numpy()
            true_labels = true_labels.data.cpu().numpy()
            kclusterer = KMeans(n_clusters=3, n_init=20) #, n_jobs=4)  # KMeans(4,n_init=20)
            y_pred = kclusterer.fit_predict(to_eval)

            # y_pred = np.asarray(y_pred)

            # print(' ' * 8 + '|==>  acc: %.4f,  nmi: %.4f  <==|'
            #       % (self.acc(true_labels, y_pred), nmi(true_labels, y_pred)))

            # currentAcc = self.acc(true_labels, y_pred)

            final_index_list = [item for sublist in final_index_list for item in sublist]

            for j in range(len(final_index_list)):
                index_length.append(len(index_list))
                index_dict.update({final_index_list[j]: y_pred[j]})

        return y_pred, index_dict

    def Feature_extractor(self, model, train_loader, word2id, id2word, train_sentence_sequence, num_words_train_set,
                          num_words_valid_set, supv_unsupv, test_loader, train_test):

        optimizer = optim.SGD(model.parameters(), lr=constant.params["lr"], weight_decay=0.01, momentum=0.9)

        optimizer = optim.Adam(
            model.parameters(),
            betas=(constant.params["optimizer_adam_beta1"],
                   constant.params["optimizer_adam_beta2"]),
            lr=constant.params["lr"])

        lr_scheduler_step, lr_scheduler_epoch = None, None  # lr schedulers
        lrd_scheme, lrd_range = constant.params["lr_decay"].split('_')
        lrd_func = lr_decay_map()[lrd_scheme]

        lr_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lrd_func(constant.params),
            last_epoch=int(model.iterations) or -1
        )

        lr_scheduler_epoch, lr_scheduler_step = None, None
        if lrd_range == 'epoch':
            lr_scheduler_epoch = lr_scheduler
        elif lrd_range == 'step':
            lr_scheduler_step = lr_scheduler
        else:
            raise ValueError("Unknown lr decay range {}".format(lrd_range))

        print("Parameters: {}(trainable), {}(non-trainable)".format(
            compute_num_params(model)[0], compute_num_params(model)[1]))
        criterion = cust1()
        criterion2 = cust2()

        start_iter = 0

        best_acc = 0.0

        for epoch in range(constant.params["num_epochs"]):  # constant.params["num_epochs"]

            sys.stdout.flush()
            log_loss = []
            running_loss = 0
            correct = 0
            total_train = 0
            correct_train = 0
            all_embs = []
            dic = {}
            loss = 0
            count = 0
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            max_values = []
            model.train()

            for i, (data) in enumerate(pbar, start=start_iter):
                word_x, x_len, raw_x, sentence, sentence_lengths, index_list, word_y, y_len, sentence_word, sentence_word_lengths, _, _ = data  # word_x, char_x, x_len, raw_x, sentence,
                max_target = []
                if constant.USE_CUDA:
                    sentence = sentence.cuda()
                    sentence_word = sentence_word.cuda()
                else:
                    sentence = sentence.to(device)
                    sentence_word = sentence_word.to(device)
                optimizer.zero_grad()

                output, X_out = model(sentence, sentence_lengths, supv_unsupv, sentence_word, sentence_word_lengths,
                                      train_test)  # ,word_x  #,X_word_label

                loss1 = criterion(output)  # ,X_word_label
                loss2 = criterion2(output)
                loss = loss2
                running_loss += loss.item()
                log_loss.append(loss.item())
                a = list(model.parameters())[0].clone()
                list(model.parameters())[0].grad
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

                if lr_scheduler_step:
                    lr_scheduler_step.step()

                b = list(model.parameters())[0].clone()
                if torch.equal(a.data, b.data):
                    count = count + 1

            if lr_scheduler_epoch:
                lr_scheduler_epoch.step()
            torch.save(model, 'bestModel'.format(best_acc))

            print("The loss is ", np.mean(log_loss))

            index_dict, _, _ = self.predict(model, train_loader, word2id, id2word, train_sentence_sequence,
                                            num_words_train_set, len(train_sentence_sequence), supv_unsupv, test_loader)

        return np.mean(log_loss)

    def acc(self, y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        from sklearn.utils.linear_assignment_ import linear_assignment
        ind = linear_assignment(w.max() - w)
        for i, j in ind:
            a = w[i, j]
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def predict_supv(self, model, train_loader, word2id, id2word, train_sentence_sequence, num_words_train_set,
                     num_words_valid_set, supv_unsupv, test_loader, train_test):

        model.eval()
        start_iter = 0
        index_dict = {}
        running_loss = 0
        maxx = []
        index_length = []
        log_loss = []
        running_loss = 0
        correct = 0
        total_train = 0
        correct_train = 0
        all_embs = []
        dic = {}
        if constant.USE_CUDA:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss().to(device)
        pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        with torch.no_grad():
            for i, (data) in enumerate(pbar, start=start_iter):
                word_x, x_len, raw_x, sentence, sentence_lengths, index_list, word_y, y_len, sentence_word, sentence_word_lengths, overwrite_label, overwrite_label_len = data  # word_x, char_x, x_len, raw_x, sentence,
                if constant.USE_CUDA:
                    word_x = word_x.cuda()
                    sentence = sentence.cuda()
                    sentence_word = sentence_word.cuda()
                    word_y = word_y.cuda()
                else:
                    word_x = word_x.to(device)
                    sentence = sentence.to(device)
                    sentence_word = sentence_word.to(device)
                    word_y = word_y.to(device)

                if word_y.size()[0] == 1:
                    word_y = word_y.squeeze(1)
                else:
                    word_y = word_y.squeeze()
                output, XX = model(sentence, sentence_lengths, supv_unsupv, sentence_word, sentence_word_lengths,
                                   train_test)  # ,word_x ,X_word_label

                loss = criterion(output, word_y)
                running_loss += loss.item()
                log_loss.append(loss.item())
                _, predicted = torch.max(output.data, 1)
                total_train += word_y.nelement()
                correct_train += predicted.eq(word_y.data).sum().item()
            accuracy = 100 * correct_train / total_train

            print("Accuracy = {}".format(accuracy))

        return log_loss, accuracy
