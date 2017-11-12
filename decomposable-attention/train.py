import sys
import os
sys.path.append(os.pardir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from utilities.data_loader import *
import model


hidden_size = 100
learning_rate = 0.05
batch_size = 64
weight_decay = 5e-5


def test_model(loader, input_encoder, inter_atten, use_cuda, num_batch=100):
    correct = 0
    total = 0
    input_encoder.eval()
    inter_atten.eval()

    for _ in range(100):
        judgement, question_1, question_2 = next(loader)
        
        if use_cuda:
            question_1_var = Variable(torch.LongTensor(question_1).cuda())
            question_2_var = Variable(torch.LongTensor(question_2).cuda())
            judgement_var = Variable(torch.FloatTensor(judgement).cuda())
        else:
            question_1_var = Variable(torch.LongTensor(question_1))
            question_2_var = Variable(torch.LongTensor(question_2))
            judgement_var = Variable(torch.FloatTensor(judgement))

        embed_1, embed_2 = input_encoder(question_1_var, question_2_var)
        prob = inter_atten(embed_1, embed_2).squeeze()

        predict = np.array(prob.cpu().data.numpy() > 0.5, dtype=np.int0)
        labet = np.array(judgement, dtype=np.int0)
        total += len(judgement)
        correct += np.sum(predict == labet)
            
    input_encoder.train()
    inter_atten.train()

    return (correct / total)


def train(max_batch):
    data_dir = '../data'
    vocabulary, word_embeddings, word_to_index_map, index_to_word_map = load_embed(data_dir + '/wordvec.txt')
    training_set = load_data(data_dir + '/train.tsv', word_to_index_map)[:1600]  # subset for faster test
    train_iter = batch_iter(training_set, batch_size)

    dev_set = load_data(data_dir + '/dev.tsv', word_to_index_map)
    dev_iter = batch_iter(dev_set, 100)

    use_cuda = torch.cuda.is_available()

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

    criterion = nn.BCELoss()

    losses = []

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
        prob = inter_atten(embed_1, embed_2).squeeze()
        loss = criterion(prob, judgement_var)
        losses.append(loss.data[0])
        loss.backward()
        
        input_optimizer.step()
        inter_atten_optimizer.step()

        if (i + 1) % 100 == 0:
            train_acc = test_model(train_iter, input_encoder, inter_atten, use_cuda)
            dev_acc = test_model(dev_iter, input_encoder, inter_atten, use_cuda)
            
            print('batches %d, train-loss %.3f, train-acc %.3f, dev-acc %.3f' %
                  (i + 1, sum(losses) / len(losses), train_acc, dev_acc))
            losses = []

        if i == max_batch:
            return


if __name__ == '__main__':
    train(1e7)
