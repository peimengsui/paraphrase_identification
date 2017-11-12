'''
simple baseline model 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class baseline(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_class = 2, addBias = False):
        super(baseline, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, self.embed_size, padding_idx = 1)
        self.linear = nn.Linear(self.embed_size, self.hidden_size, bias = addBias)
        self.linear.weight.data.normal_(0, 0.01)
        if addBias:
            self.linear.bias.data.uniform_(-0.01, 0.01)
        self.final = nn.Linear(self.hidden_size, num_class)
        self.log_prob = nn.Logsoftmax()
        
    def forward(self, sent1, sent2):
        batch_size = sent1.size(0)
        # Embed input sentence pair
        sent1 = self.embedding(sent1).view(-1, self.embed_size)
        sent2 = self.embedding(sent2).view(-1, self.embed_size)
        # Feed forward layer
        sent1 = self.linear(sent1).view(batch_size, -1, self.hidden_size)
        sent1 = F.relu(sent1)
        sent2 = self.linear(sent2).view(batch_size, -1, self.hidden_size)
        sent2 = F.relu(sent2)
        ret = self.final(torch.cat((sent1, sent2), 2))
        log_prob = self.log_prob(ret)
        return log_prob
        
        
        
        
        
        