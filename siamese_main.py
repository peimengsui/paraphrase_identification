from utilities import data_loader
import torch
import torch.autograd as ta
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from siamese_lstm import Siamese
from torch.autograd import Variable
import numpy as np

data_dir = 'data'
vocabulary, word_embeddings, word_to_index_map, index_to_word_map = data_loader.load_embed(data_dir + '/wordvec.txt')
training_set = data_loader.load_data(data_dir + '/train.tsv', word_to_index_map)
valid_set = data_loader.load_data(data_dir + '/dev.tsv', word_to_index_map)

train_loader = data_loader.batch_iter(training_set, 32)
valid_loader = data_loader.batch_iter(valid_set, 32)

def test(batch_size, model, data_iter, criterion, test_set, epoch):
    step = 1
    losses = []
    correct = 0
    total_batches = int(len(test_set) / batch_size)
    while True:
        model.eval()
        label, sentence_a_words, sentence_b_words = next(data_iter) 
        label = Variable(torch.FloatTensor(label))
        sentence_a_words = Variable(torch.LongTensor(sentence_a_words))
        sentence_b_words = Variable(torch.LongTensor(sentence_b_words))
        if torch.cuda.is_available():
            label = label.cuda()
            sentence_a_words = sentence_a_words.cuda()
            sentence_b_words = sentence_b_words.cuda()
        output = model(sentence_a_words, sentence_b_words)
        pred = (output.data>0.5).float()
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss = criterion(output, label)
        losses.append(loss.data[0])
        if step % total_batches == 0:
            print( "Epoch:", (epoch), "Avg test Loss:", np.mean(losses), "Test Accuracy:", correct/(batch_size*step))
            correct = 0
            break
        step += 1

def training_loop(batch_size, model, optim, train_loader, valid_loader, num_epochs, criterion, training_set, valid_set):
    step = 1
    epoch = 1
    losses = []
    correct = 0
    batch_idx = 1
    total_batches = int(len(training_set) / batch_size)
    while epoch <= num_epochs:
        model.train()
        label, sentence_a_words, sentence_b_words = next(train_loader) 
        label = Variable(torch.FloatTensor(label))
        sentence_a_words = Variable(torch.LongTensor(sentence_a_words))
        sentence_b_words = Variable(torch.LongTensor(sentence_b_words))
        if torch.cuda.is_available():
            label = label.cuda()
            sentence_a_words = sentence_a_words.cuda()
            sentence_b_words = sentence_b_words.cuda()
        model.zero_grad()
        output = model(sentence_a_words, sentence_b_words)
        pred = (output.data>0.5).float()
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss = criterion(output, label)
        losses.append(loss.data[0])
        loss.backward()
        optim.step()
        if batch_idx % print_freq == 0:
            print( "Epoch:", (epoch), "Avg Loss:", np.mean(losses), "Accuracy:", correct/(batch_size*batch_idx))
        batch_idx += 1
        if step % total_batches == 0:
            test(batch_size, model, valid_loader, criterion, valid_set, epoch)
            correct = 0
            batch_idx = 1
            losses = []
            epoch += 1
        step += 1

num_epochs = 10
print_freq = 100
model = Siamese(word_embeddings)
if torch.cuda.is_available():
    model.cuda()
para1 = filter(lambda p: p.requires_grad, model.parameters())
optim = opt.Adam(para1)
criterion = torch.nn.BCELoss()
training_loop(32, model, optim, train_loader, valid_loader, num_epochs, criterion, training_set, valid_set)
