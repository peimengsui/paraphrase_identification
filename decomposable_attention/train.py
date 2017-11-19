import sys
import os
sys.path.append(os.pardir)

from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from utilities.data_loader import load_data, load_embed, batch_iter, add_char_ngrams
import model
from eval import test_model

hidden_size = 200
learning_rate = 0.01
batch_size = 64
weight_decay = 5e-5
max_grad_norm = 5.0
threshould = 0.5
use_ngram = True


def train(max_batch):
    start_time = datetime.now().strftime(r'%m%d_%H%M')

    data_dir = '../data'
    vocabulary, word_embeddings, word_to_index_map, index_to_word_map = load_embed(data_dir + '/wordvec.txt')
    word_embeddings = (word_embeddings.T / np.linalg.norm(word_embeddings, axis=1)).T
    training_set = load_data(data_dir + '/train.tsv', word_to_index_map, add_reversed=True, n=5)
    print('training set loaded')
    if use_ngram:
        training_set, ngram_to_index_map, index_to_ngram_map = add_char_ngrams(training_set, build_ngram_map=True, ngram_to_index_map=None)
        print('training set ngrams added')
    train_iter = batch_iter(training_set, batch_size, use_ngram)

    dev_set = load_data(data_dir + '/dev.tsv', word_to_index_map)
    if use_ngram:
        dev_set = add_char_ngrams(training_set, build_ngram_map=False, ngram_to_index_map=ngram_to_index_map)
        print('dev set ngrams added')
    dev_iter = batch_iter(dev_set, 100, use_ngram)

    use_cuda = torch.cuda.is_available()

    if use_ngram:
        input_encoder = model.encoder_char(len(ngram_to_index_map.keys()), embedding_size=300, hidden_size=hidden_size, para_init=0.01, padding_index=1)
    else:
        input_encoder = model.encoder(word_embeddings.shape[0], embedding_size=300, hidden_size=hidden_size, para_init=0.01, padding_index=1)
        input_encoder.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
        input_encoder.embedding.weight.requires_grad = False
    inter_atten = model.atten(hidden_size=hidden_size, label_size=1, para_init=0.01)

    if use_cuda:
        input_encoder.cuda()
        inter_atten.cuda()

    para1 = list(filter(lambda p: p.requires_grad, input_encoder.parameters()))
    para2 = list(inter_atten.parameters())

    input_optimizer = optim.Adagrad(para1, lr=learning_rate, weight_decay=weight_decay)
    inter_atten_optimizer = optim.Adagrad(para2, lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.BCEWithLogitsLoss()

    losses = []
    best_acc = 0
    total = 0
    correct = 0

    for i, (judgement, question_1, question_2) in enumerate(train_iter):
        input_encoder.train()
        inter_atten.train()

        if use_cuda:
            question_1_var = Variable(torch.LongTensor(question_1).cuda())
            question_2_var = Variable(torch.LongTensor(question_2).cuda())
            judgement_var = Variable(torch.FloatTensor(judgement).cuda())
        else:
            question_1_var = Variable(torch.LongTensor(question_1))
            question_2_var = Variable(torch.LongTensor(question_2))
            judgement_var = Variable(torch.FloatTensor(judgement))
            
        input_encoder.zero_grad()
        inter_atten.zero_grad()

        embed_1, embed_2 = input_encoder(question_1_var, question_2_var)  # batch_size * length * embedding_dim
        logits, prob = inter_atten(embed_1, embed_2)
        loss = criterion(logits, judgement_var)
        losses.append(loss.data[0])
        loss.backward()

        predict = np.array(prob.cpu().data.numpy() > threshould, dtype=np.int0)
        label = np.array(judgement, dtype=np.int0)
        total += len(judgement)
        correct += np.sum(predict == label)
        
        grad_norm = 0.
        para_norm = 0.

        for m in input_encoder.modules():
            if isinstance(m, nn.Linear):
                grad_norm += m.weight.grad.data.norm() ** 2
                para_norm += m.weight.data.norm() ** 2
                if m.bias is not None:
                    grad_norm += m.bias.grad.data.norm() ** 2
                    para_norm += m.bias.data.norm() ** 2

        for m in inter_atten.modules():
            if isinstance(m, nn.Linear):
                grad_norm += m.weight.grad.data.norm() ** 2
                para_norm += m.weight.data.norm() ** 2
                if m.bias is not None:
                    grad_norm += m.bias.grad.data.norm() ** 2
                    para_norm += m.bias.data.norm() ** 2

        grad_norm ** 0.5
        para_norm ** 0.5

        shrinkage = max_grad_norm / (grad_norm+1e-6)
        if shrinkage < 1:
            for m in input_encoder.modules():
                if isinstance(m, nn.Linear):
                    m.weight.grad.data = m.weight.grad.data * shrinkage
            for m in inter_atten.modules():
                if isinstance(m, nn.Linear):
                    m.weight.grad.data = m.weight.grad.data * shrinkage
                    m.bias.grad.data = m.bias.grad.data * shrinkage

        input_optimizer.step()
        inter_atten_optimizer.step()

        if (i + 1) % 100 == 0:
            train_acc = test_model(train_iter, input_encoder, inter_atten, use_cuda)
            dev_acc = test_model(dev_iter, input_encoder, inter_atten, use_cuda)

            if dev_acc > best_acc:
                best_acc = dev_acc
                torch.save(input_encoder.state_dict(), 'input_encoder'+start_time+'.pt')
                torch.save(inter_atten.state_dict(), 'inter_atten'+start_time+'.pt')
           
            print('batches %d, train-loss %.3f, train-acc %.3f, dev-acc %.3f, best-dev-acc %.3f, grad-norm %.3f, para-norm %.3f' %
                  (i + 1, sum(losses) / len(losses), correct / total, dev_acc, best_acc, grad_norm, para_norm))

            losses = []
            total = 0
            correct = 0

        if i == max_batch:
            return


if __name__ == '__main__':
    train(1e7)
