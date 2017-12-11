import sys
import os
sys.path.append(os.pardir)

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from utilities.data_loader import *
from baseline import baseline
import argparse
import time
import pickle as pkl
import numpy as np

# Set hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='../data/', help='Path to data folder')
parser.add_argument("--embed_file", type=str, default='wordvec.txt', help='pre-trained word embedding')
parser.add_argument("--train_file", type=str, default='train.tsv', help='name of training data file')
parser.add_argument("--valid_file", type=str, default='dev.tsv', help='name of validation data file')
parser.add_argument("--test_file", type=str, default='test.tsv', help='name of testing data file')
parser.add_argument("--n_gram", type=int, default=0, help='n for n_gram')
parser.add_argument("--rpt_step", type=int, default=1000, help='Report training loss and accuracy every n step')
parser.add_argument("--val_step", type=int, default=10000, help='Report valid loss and accuracy every n step')
parser.add_argument("--resume_model", type=str, default=None, help='File name of model to resume')
parser.add_argument("--num_step", type=int, default=10000000, help='Total number of step for training')
parser.add_argument("--batch_size", type=int, default=32, help='Batch size for training')
parser.add_argument("--test_size", type=int, default=200, help='Batch size for testing')
parser.add_argument("--lr", type=float, default=0.05, help='Learning rate')
parser.add_argument("--fine_tune", type=bool, default=False, help='Whether to fine tune embedding')
parser.add_argument("--emb_dim", type=int, default=300, help='Embedding dimension')
parser.add_argument("--hid_dim", type=int, default=200, help='Hidden dimension')
args = parser.parse_args()


# Load data
vocabulary, word_embeddings, word2id, id2word = load_embed(args.data_dir + args.embed_file, embedding_dim = args.emb_dim)
train_set = load_data(args.data_dir + args.train_file, word2id, add_reversed=True, n=args.n_gram)
valid_set = load_data(args.data_dir + args.valid_file, word2id, add_reversed=True, n=args.n_gram)
test_set = load_data(args.data_dir + args.test_file, word2id, n=args.n_gram)
vocab_size = len(id2word)


train_batches = batch_iter(train_set, args.batch_size, shuffle=True)
valid_batches = batch_iter(valid_set, args.batch_size * 10, shuffle=True)
test_batches = batch_iter(test_set, args.test_size, shuffle=False)

def set_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    return var

model = baseline(vocab_size, args.emb_dim, args.hid_dim)
# Initialize with pre-trained word_embedding. Otherwise train wordembedding from scratch
if args.embed_file is not None:
    model.embedding.weight.data.copy_(torch.FloatTensor(word_embeddings))

if not args.fine_tune:
    model.embedding.weight.requires_grad = False

model = set_cuda(model)

if args.resume_model is not None:
    model.load_state_dict(torch.load(args.resume_model))

para = filter(lambda p: p.requires_grad, model.parameters())
opt = optim.Adagrad(para, lr=args.lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
# Start training
step = 0
best_acc = 0
for train_batch in train_batches:
    step += 1
    if step == args.num_step:
        print('Finish training in %.3f seconds' % (time.time() - start))
        break
    opt.zero_grad()
    labels, sents1, sents2 = train_batch
    labels = set_cuda(Variable(torch.LongTensor(labels)))
    sents1 = set_cuda(Variable(torch.LongTensor(sents1)))
    sents2 = set_cuda(Variable(torch.LongTensor(sents2)))
    log_prob = model(sents1, sents2)
    loss = criterion(log_prob, labels)
    loss.backward()
    opt.step()
    if step % args.rpt_step == 0:
        _, predicts = log_prob.data.max(dim = 1)
        total = len(labels)
        correct = torch.sum(predicts == labels.data)
        acc = float(correct / total)
        print('Training: step: %d, avg loss: %.3f, acc: %.3f' % (step, loss.data[0], acc))
    if step % args.val_step == 0:
        labels, sents1, sents2 = next(valid_batches)
        labels = set_cuda(Variable(torch.LongTensor(labels)))
        sents1 = set_cuda(Variable(torch.LongTensor(sents1)))
        sents2 = set_cuda(Variable(torch.LongTensor(sents2)))
        log_prob = model(sents1, sents2)
        loss = criterion(log_prob, labels)
        _, predicts = log_prob.data.max(dim = 1)
        total = len(labels)
        correct = torch.sum(predicts == labels.data)
        acc = float(correct / total)
        print('Validation: step: %d, avg loss: %.3f, acc: %.3f, best acc: %.3f' % (step, loss.data[0], acc, best_acc))
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'baseline.pt')
# Final test on the whole test set
acc = []
for i in range(len(test_set) / args.test_size):
    labels, sents1, sents2 = next(test_batches)
    labels = set_cuda(Variable(torch.LongTensor(labels)))
    sents1 = set_cuda(Variable(torch.LongTensor(sents1)))
    sents2 = set_cuda(Variable(torch.LongTensor(sents2)))
    log_prob = model(sents1, sents2)
    loss = criterion(log_prob, labels)
    _, predicts = log_prob.data.max(dim = 1)
    acc += (predicts == labels.data).tolist()

print('Test loss: %.3f, test acc: %.3f' % (loss.data[0], np.mean(acc)))
f = open('acc_record.pkl','wb')
pkl.dump(acc, f)
f.close()
        

            
    
    
    


