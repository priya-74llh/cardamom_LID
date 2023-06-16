 
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
#from src.model import MetaEmbeddingCLUSTER
from src import constant

def create_emb_layer(train_test,weight_matrix, padding_idx, embedding_dim,num_embeddings,non_trainable=False):
    
    emb_layer = nn.Embedding(num_embeddings, embedding_dim,padding_idx)
    if train_test == "train":
        emb_layer.load_state_dict({'weight': weight_matrix})
        if non_trainable:
            emb_layer.weight.requires_grad = True
    return emb_layer, num_embeddings, embedding_dim

class NgramModel(nn.Module):
    def __init__(self,train_test,nb_layers,embedding_dimension,nb_lstm_units,batch_size,vocab,weight_matrix):
        super(NgramModel, self).__init__()
        
        self.iterations = 0
        self.nb_layers = nb_layers
        self.embedding_dimension = embedding_dimension
        self.nb_lstm_units = nb_lstm_units
        self.batch_size = batch_size
        self.vocab = vocab 
        
        nb_vocab_words = len(self.vocab)


         # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab['<pad>']

        self.embedding, num_embeddings, embedding_dim = create_emb_layer(train_test,weight_matrix, padding_idx,embedding_dimension,nb_vocab_words,True)

        self.lstm = nn.LSTM(
            input_size=self.embedding_dimension,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_layers,
            batch_first=True,bidirectional=True
        )

        # output layer which projects back to tag space
        self.hidden_to_LID = nn.Linear(self.nb_lstm_units*2, 128)
        self.output = nn.Linear(128, 11)


    
    def init_hidden(self,batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_layers*2, batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_layers*2, batch_size, self.nb_lstm_units)

        if constant.USE_CUDA:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence

        batch_size, seq_len = X.size()
        
        self.hidden = self.init_hidden(batch_size)

        

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X = self.embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, (h_t, h_c) = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        out1, out2 = torch.chunk(X, 2, dim=2)
        out_cat = torch.cat((out1[:, -1, :], out2[:, 0, :]), 1)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        # X = X.contiguous()
        # X = X.view(-1, X.shape[2])

        h_t = h_t.view(-1, h_t.shape[2])

        # run through actual linear layer
        X = F.relu(self.hidden_to_LID(out_cat))

        X = F.relu(self.output(X))

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.softmax(X, dim=1)

        

        Y_hat = X
        return Y_hat