import sys
import os
sys.path.append(os.pardir)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utilities.data_loader import *
from baseline import baseline

# Load data
data_dir = '../data'
vocabulary, word_embeddings, word2id, id2word = load_embed(data_dir + '/wordvec.txt')
train_set = load_data(data_dir + '/train.tsv', word2id)
valid_set = load_data(data_dir + '/dev.tsv', word2id)
test_set = load_data(data_dir + '/test.tsv', word2id)

batch_size = 32
vocab_size = len(id2word)
embed_size = 300
hidden_size = 200
learning_rate = 0.05
num_epoch = 50000000
train_batches = batch_iter(train_set, batch_size)
valid_batches = batch_iter(valid_set, batch_size * 10)
test_batches = batch_iter(test_set, len(test_set))
    
model = baseline(vocab_size, embed_size, hidden_size)
# Initialize and fix word embedding with glove vector
model.embedding.weight.data.copy_(torch.FloatTensor(word_embeddings))
model.embedding.weight.requires_grad = False
model.cuda()

para = filter(lambda p: p.requires_grad, model.parameters())
opt = optim.Adagrad(para, lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Start training
epoch = 0
best_acc = 0
for train_batch in train_batches:
    epoch += 1
    if epoch == num_epoch:
        break
    opt.zero_grad()
    labels, sents1, sents2 = train_batch
    labels = Variable(torch.LongTensor(labels)).cuda()
    sents1 = Variable(torch.LongTensor(sents1)).cuda()
    sents2 = Variable(torch.LongTensor(sents2)).cuda()
    log_prob = model(sents1, sents2)
    loss = criterion(log_prob, labels)
    loss.backward()
    opt.step()
    if epoch % 1000 == 0:
        _, predicts = log_prob.data.max(dim = 1)
        total = len(labels)
        correct = torch.sum(predicts == labels.data)
        acc = float(correct / total)
        print('Training: epoch: %d, avg loss: %.3f, acc: %.3f' % (epoch, loss.data[0], acc))
    if epoch % 10000 == 0:
        labels, sents1, sents2 = next(valid_batches)
        labels = Variable(torch.LongTensor(labels)).cuda()
        sents1 = Variable(torch.LongTensor(sents1)).cuda()
        sents2 = Variable(torch.LongTensor(sents2)).cuda()
        log_prob = model(sents1, sents2)
        loss = criterion(log_prob, labels)
        _, predicts = log_prob.data.max(dim = 1)
        total = len(labels)
        correct = torch.sum(predicts == labels.data)
        acc = float(correct / total)
        print('Validation: epoch: %d, avg loss: %.3f, acc: %.3f, best acc: %.3f' % (epoch, loss.data[0], acc, best_acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'baseline.pt')
# Final test
labels, sents1, sents2 = next(test_batches)
labels = Variable(torch.LongTensor(labels)).cuda()
sents1 = Variable(torch.LongTensor(sents1)).cuda()
sents2 = Variable(torch.LongTensor(sents2)).cuda()
log_prob = model(sents1, sents2)
loss = criterion(log_prob, labels)
_, predicts = log_prob.data.max(dim = 1)
total = len(labels)
correct = torch.sum(predicts == labels.data)
acc = float(correct / total)
print('Testing: epoch: %d, avg loss: %.3f, acc: %.3f' % (epoch, loss.data[0], acc, best_acc))
        

            
    
    
    


