import sys
import os
sys.path.append(os.pardir)
from utilities import data_loader
import torch
import torch.autograd as ta
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from siamese_lstm import Siamese
from torch.autograd import Variable
import numpy as np

data_dir = '../data'
vocabulary, word_embeddings, word_to_index_map, index_to_word_map = data_loader.load_embed(data_dir + '/wordvec.txt')
training_set = data_loader.load_data(data_dir+'/train.tsv', word_to_index_map)
valid_set = data_loader.load_data(data_dir+'/dev.tsv', word_to_index_map)
test_set = data_loader.load_data(data_dir+'/test.tsv', word_to_index_map)

train_loader = data_loader.batch_iter(training_set, 50)
valid_loader = data_loader.batch_iter(valid_set, 50)
test_loader = data_loader.batch_iter(test_set, 50)

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
            print( "Epoch:", (epoch), "Avg test Loss:", np.mean(losses), "Test Accuracy:", correct/(batch_size*total_batches))
            return correct/(batch_size*total_batches) 
        step += 1

def training_loop(batch_size, model, optim, train_loader, valid_loader, num_epochs, criterion, training_set, valid_set):
    step = 1
    epoch = 1
    losses = []
    correct = 0
    best_acc = 0
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
        if step % total_batches == 0:
            print( "Epoch:", (epoch), "Training Avg Loss:", np.mean(losses), "Training Accuracy:", correct/(batch_size*total_batches))
            acc = test(batch_size, model, valid_loader, criterion, valid_set, epoch)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_siamese.pt')
            correct = 0
            losses = []
            epoch += 1
        step += 1

num_epochs = 50
model = Siamese(word_embeddings)
if torch.cuda.is_available():
    model.cuda()
para1 = filter(lambda p: p.requires_grad, model.parameters())
optim = opt.Adam(para1)
criterion = torch.nn.BCELoss()
training_loop(50, model, optim, train_loader, valid_loader, num_epochs, criterion, training_set, valid_set)
print('Final Accuracy Result on Test Dataset /n')
model = Siamese(word_embeddings)
model.load_state_dict(torch.load('best_siamese.pt'))
if torch.cuda.is_available():
    model.cuda()
test(50, model, test_loader, criterion, test_set, num_epochs)
