import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.metrics import f1_score
import time
import math
from sklearn.model_selection import GroupKFold
import sys

def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight) 
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)


class AttentionWordRNN(nn.Module):
    def __init__(self, batch_size, num_tokens, embed_size, word_gru_hidden, bidirectional= True):        
        
        super(AttentionWordRNN, self).__init__()
        
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        
        self.lookup = nn.Embedding(num_tokens, embed_size)
        if bidirectional == True:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= True)
            self.weight_W_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,2*word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(2* word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(2*word_gru_hidden, 1))
        else:
            self.word_gru = nn.GRU(embed_size, word_gru_hidden, bidirectional= False)
            self.weight_W_word = nn.Parameter(torch.Tensor(word_gru_hidden, word_gru_hidden))
            self.bias_word = nn.Parameter(torch.Tensor(word_gru_hidden,1))
            self.weight_proj_word = nn.Parameter(torch.Tensor(word_gru_hidden, 1))
            
        self.softmax_word = nn.Softmax()
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)
        
    def forward(self, embed, state_word):
        # embeddings
        embedded = self.lookup(embed)
        # word level gru
        output_word, state_word = self.word_gru(embedded, state_word)
        word_squish = batch_matmul_bias(output_word, self.weight_W_word,self.bias_word, nonlinearity='tanh')
        word_attn = batch_matmul(word_squish, self.weight_proj_word)
        word_attn_norm = self.softmax_word(word_attn.transpose(1,0))
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1,0))        
        return word_attn_vectors, state_word, word_attn_norm
    
    def init_hidden(self):
        if self.bidirectional == True:
            return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden)).cuda()
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden)).cuda() 


class LinearLayer(nn.Module):
    def __init__(self, word_gru_hidden, n_classes, bidirectional= True):        
        super(LinearLayer, self).__init__() 
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional

        if bidirectional == True:
            self.final_linear = nn.Linear(2*word_gru_hidden, n_classes)
        else:
            self.final_linear = nn.Linear(word_gru_hidden, n_classes)

    def forward(self, word_attention_vectors):
        pooled, _ = torch.max(word_attention_vectors, 0)
        final_map = self.final_linear(pooled.squeeze(0))
        return F.log_softmax(final_map)


def train_data(mini_batch, targets, word_attn_model, linear_layer, word_optimizer, linear_optimizer, criterion):
    state_word = word_attn_model.init_hidden()
    max_sents, batch_size, max_tokens = mini_batch.size()
    word_optimizer.zero_grad()
    linear_optimizer.zero_grad()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred = linear_layer(s)

    loss = criterion(y_pred, targets) 
    loss.backward()
    
    torch.nn.utils.clip_grad_norm(word_attn_model.parameters(), 0.2)
    torch.nn.utils.clip_grad_norm(linear_layer.parameters(), 0.2)

    word_optimizer.step()
    linear_optimizer.step()
    return loss.data.cpu()

def get_predictions(tokens, word_attn_model, linear_layer):        
    max_sents, batch_size, max_tokens = tokens.size()
    state_word = word_attn_model.init_hidden()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(tokens[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred = linear_layer(s)    
    return y_pred

def pad_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(x) for x in mini_batch]))
    max_token_len = int(np.max([len(val) for sublist in mini_batch for val in sublist]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype= np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i,j,k] = mini_batch[i][j][k]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix).transpose(0,1)).cuda()

def test_accuracy_mini_batch(token, labels, word_attn, linear_layer):
    y_pred = get_predictions(token, word_attn, linear_layer)
    _, y_pred = torch.max(y_pred, 1)
    correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
    labels = np.ndarray.flatten(labels.data.cpu().numpy())
    num_correct = sum(correct == labels)
    return float(num_correct) / len(correct)

def test_accuracy_full_batch(tokens, labels, mini_batch_size, word_attn, linear_layer):
    p = []
    l = []
    g = gen_minibatch(tokens, labels, mini_batch_size)
    for token, label in g:
        y_pred = get_predictions(token, word_attn, linear_layer)
        _, y_pred = torch.max(y_pred, 1)
        p.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        l.append(np.ndarray.flatten(label.data.cpu().numpy()))
    p = [item for sublist in p for item in sublist]
    l = [item for sublist in l for item in sublist]
    p = np.array(p)
    l = np.array(l)
    num_correct = sum(p == l)
    print(l)
    print(p)
    return (float(num_correct)/ len(p), f1_score(l, p))

def test_data(mini_batch, targets, word_attn_model, linear_layer, criterion):    
    state_word = word_attn_model.init_hidden()
    max_sents, batch_size, max_tokens = mini_batch.size()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(mini_batch[i,:,:].transpose(0,1), state_word)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)            
    y_pred = linear_layer(s)
    loss = criterion(y_pred, targets)     
    return loss.data.cpu()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def gen_minibatch(tokens, labels, mini_batch_size, shuffle= False):
    for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle= shuffle):
        yield pad_batch(token), Variable(torch.from_numpy(label).cuda(), requires_grad= False)

def check_val_loss(val_tokens, val_labels, mini_batch_size, word_attn_model, linear_layer, criterion):
    val_loss = []
    for token, label in iterate_minibatches(val_tokens, val_labels, mini_batch_size, shuffle= True):
        val_loss.append(test_data(pad_batch(token), Variable(torch.from_numpy(label).cuda(), requires_grad= False), 
                                  word_attn_model, linear_layer, criterion))
    return np.mean(val_loss)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train_early_stopping(fold, mini_batch_size, X_train, y_train, X_test, y_test, word_attn_model, linear_layer, 
                         word_attn_optimiser, linear_optimiser, loss_criterion, num_epoch, 
                         print_val_loss_every = 10):
    start = time.time()
    loss_full = []
    loss_epoch = []
    accuracy_epoch = []
    loss_smooth = []
    accuracy_full = []
    epoch_counter = 0
    g = gen_minibatch(X_train, y_train, mini_batch_size)
    min_val_loss = 1000
    min_idx = -1
    for i in range(1, num_epoch + 1):
        try:
            tokens, labels = next(g)
            loss = train_data(tokens, labels, word_attn_model, linear_layer, word_attn_optimiser, linear_optimiser, loss_criterion)
            acc = test_accuracy_mini_batch(tokens, labels, word_attn_model, linear_layer)
            accuracy_full.append(acc)
            accuracy_epoch.append(acc)
            loss_full.append(loss)
            loss_epoch.append(loss)
            # check validation loss every n passes
            if i % print_val_loss_every == 0:
                val_loss = check_val_loss(X_test, y_test, mini_batch_size, word_attn_model, linear_layer, loss_criterion)
                if np.isnan(val_loss):
                    return None
                print('Loss at %d minibatches, %d epoch,(%s) is %f' %(i, epoch_counter, timeSince(start), np.mean(loss_epoch)))
                print('Accuracy at %d minibatches is %f' % (i, np.mean(accuracy_epoch)))
                print('Validation loss after %d passes is %f' %(i, val_loss))
                sys.stdout.flush()
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_idx = i
                    torch.save(word_attn_model.state_dict(), 'saved_models/gru/fold{}_word_attn.pth'.format(fold))
                    torch.save(linear_layer.state_dict(), 'saved_models/gru/fold{}_linear.pth'.format(fold))
        except StopIteration:
            epoch_counter += 1
            print('Reached %d epocs' % epoch_counter)
            print('i %d' % i)
            sys.stdout.flush()
            g = gen_minibatch(X_train, y_train, mini_batch_size)
            loss_epoch = []
            accuracy_epoch = []
    return min_idx

def evaluate_gru(batchsize):
    dbank = pd.read_json('dbank.json')
    X = dbank['tokens'].sample(frac=1, random_state=20)
    y = dbank['labels'].sample(frac=1, random_state=20)
    ids = dbank['ids'].sample(frac=1, random_state=20)

    group_kfold = GroupKFold(n_splits=10).split(X, y, groups=ids)
    data = []

    for train_index, test_index in group_kfold:
        fold = {}
        fold["X_train"] = X.values[train_index]
        fold["y_train"] = y.values[train_index]
        fold["X_test"]  = X.values[test_index]
        fold["y_test"]  = y.values[test_index]
        fold["train_ids"]  = np.array(ids)[train_index]

        data.append(fold)

    learning_rate = 1e-1
    momentum = 0.9
    criterion = nn.NLLLoss()
    idx = 0
    accuracy = 0
    f_measure = 0

    while idx < 10:
        fold = data[idx]
        X_train, y_train = fold["X_train"], fold["y_train"].ravel()  # Ravel flattens a (n,1) array into (n, )
        X_test, y_test   = fold["X_test"], fold["y_test"].ravel()
        split = len(X_train)//10
        X_validate = X_train[:split]
        y_validate = y_train[:split]
        X_train = X_train[split:]
        y_train = y_train[split:]

        word_attn = AttentionWordRNN(batch_size=batchsize, num_tokens=1829, embed_size=300, 
                             word_gru_hidden=100, bidirectional= True).cuda()
        linear_layer = LinearLayer(word_gru_hidden=100, n_classes=2, bidirectional= True).cuda()
        word_optmizer = torch.optim.SGD(word_attn.parameters(), lr=learning_rate, momentum= momentum)
        linear_optimizer = torch.optim.SGD(linear_layer.parameters(), lr=learning_rate, momentum= momentum)
        print("---------------- fold {} ----------------".format(idx))
        sys.stdout.flush()
        best_model = train_early_stopping(idx, batchsize, X_train, y_train, X_validate, y_validate, word_attn, linear_layer, word_optmizer, linear_optimizer, criterion, 160, 5)
        if not best_model:
            continue
        trained_word_attn = AttentionWordRNN(batch_size=batchsize, num_tokens=1829, embed_size=300,  word_gru_hidden=100, bidirectional= True).cuda()
        trained_linear_layer = LinearLayer(word_gru_hidden=100, n_classes=2, bidirectional= True).cuda()

        trained_word_attn.load_state_dict(torch.load('saved_models/gru/fold{}_word_attn.pth'.format(idx)))
        trained_linear_layer.load_state_dict(torch.load('saved_models/gru/fold{}_linear.pth'.format(idx)))
        trained_word_attn.eval()
        trained_linear_layer.eval()
        
        acc, f1 = test_accuracy_full_batch(X_test, y_test, batchsize, trained_word_attn, trained_linear_layer)
        print("Best model is {}".format(best_model))
        print("---------------- accuracy, f-measure of fold {} is {}, {} ----------------".format(idx, acc, f1))
        accuracy += acc
        f_measure += f1
        sys.stdout.flush()
        idx += 1
    print("average acc, f score = {}, {}".format(accuracy/10, f_measure/10))
