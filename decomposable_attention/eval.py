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


hidden_size = 200
learning_rate = 0.01
batch_size = 64
weight_decay = 5e-5
max_grad_norm = 5.0
resume = True
path_encoder = 'input_encoder_171113.pt'
path_atten = 'inter_atten_171113.pt'
threshould = 0.5


def test_model(loader, input_encoder, inter_atten, use_cuda, num_batch=100, threshould=0.5):
    correct = 0
    total = 0
    input_encoder.eval()
    inter_atten.eval()

    for _ in range(num_batch):
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

        predict = np.array(prob.cpu().data.numpy() > threshould, dtype=np.int0)
        label = np.array(judgement, dtype=np.int0)
        total += len(judgement)
        correct += np.sum(predict == label)
            
    input_encoder.train()
    inter_atten.train()

    return (correct / total)


def eval():
    data_dir = '../data'
    vocabulary, word_embeddings, word_to_index_map, index_to_word_map = load_embed(data_dir + '/wordvec.txt')
    training_set = load_data(data_dir + '/train.tsv', word_to_index_map)  # subset for faster test
    train_iter = batch_iter(training_set, batch_size)

    dev_set = load_data(data_dir + '/dev.tsv', word_to_index_map)
    dev_iter = batch_iter(dev_set, 100)

    test_set = load_data(data_dir + '/test.tsv', word_to_index_map)
    test_iter = batch_iter(test_set, 100)

    use_cuda = torch.cuda.is_available()

    input_encoder = model.encoder(word_embeddings.shape[0], embedding_size=300, hidden_size=hidden_size, para_init=0.01, padding_index=1)
    input_encoder.embedding.weight.requires_grad = False
    inter_atten = model.atten(hidden_size=hidden_size, label_size=1, para_init=0.01)

    if resume:
        input_encoder.load_state_dict(torch.load(path_encoder))
        inter_atten.load_state_dict(torch.load(path_atten))
        print('model resumed from', path_encoder, 'and', path_atten)
    else:
        input_encoder.embedding.weight.data.copy_(torch.from_numpy(word_embeddings))
    
    if use_cuda:
        input_encoder.cuda()
        inter_atten.cuda()

    para1 = list(filter(lambda p: p.requires_grad, input_encoder.parameters()))
    para2 = list(inter_atten.parameters())

    grad_norm = 0.
    para_norm = 0.

    for m in input_encoder.modules():
        if isinstance(m, nn.Linear):
            para_norm += m.weight.data.norm() ** 2
            if m.bias is not None:
                para_norm += m.bias.data.norm() ** 2

    for m in inter_atten.modules():
        if isinstance(m, nn.Linear):
            para_norm += m.weight.data.norm() ** 2
            if m.bias is not None:
                para_norm += m.bias.data.norm() ** 2

    para_norm ** 0.5

    train_acc = test_model(train_iter, input_encoder, inter_atten, use_cuda, threshould=threshould)
    dev_acc = test_model(dev_iter, input_encoder, inter_atten, use_cuda, threshould=threshould)
    test_acc = test_model(test_iter, input_encoder, inter_atten, use_cuda, threshould=threshould)
    
    print('train-acc %.3f, dev-acc %.3f, test-acc %.3f, para-norm %.3f' %
          (train_acc, dev_acc, test_acc, para_norm))


if __name__ == '__main__':
    eval()
