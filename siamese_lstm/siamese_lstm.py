import sys
import os
sys.path.append(os.pardir)
import torch
import torch.autograd as ta
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable

class WordLSTM(nn.Module):
    def __init__(self, word_vecs, rsize = 50):
        super(WordLSTM, self).__init__()
        #typical Pytorch cudnn LSTM declaration
        self.embedding = nn.Embedding(106687, 300, padding_idx=1) 
        self.embedding.weight.data.copy_(torch.FloatTensor(word_vecs))
        self.embedding.weight.requires_grad = False
        self.rnn = nn.LSTM(input_size=300, 
                           hidden_size=rsize, 
                           batch_first=True,
                           dropout = 0.2 
                          )

    def forward(self, x):
        #pass the embedding input sequence through the LSTM
        #h_0 is stand in for the initial hidden states, which is always 0 in this case
        h0 = Variable(torch.zeros(1, x.size()[0], 50))
        x = self.embedding(x)
        if torch.cuda.is_available():
            h0 = h0.cuda()
        fseq, (h_n, c_n) = self.rnn(x, (h0, h0))
        
        #there are several ways we can pool this, but just taking the last hidden state works fine enough for such a problem
        x = fseq[:,-1,:]
        
        return x

#A siamese network takes two different sets of data, processes them through two identical branches
#deploys some fancy merging function and minimizes a final metric between the two.
#Similar samples should have a low metric, dissimilar samples should have a large metric.
#Check out my (one and only) paper on this exact topic:
#https://www.researchgate.net/publication/304834009_Learning_Text_Similarity_with_Siamese_Recurrent_Networks
#although in this case the metric is a probability :)
class Siamese(nn.Module):
    def __init__(self, word_vecs):
        super(Siamese, self).__init__()
        self.word_lstm_encoder = WordLSTM(word_vecs)
        #The final classifier is a typical feedforward NN with one categorical output
    def forward(self, x, y):
        
        #we take two identical branches of parameters and feed them two different datas
        #first on the word lstm branches
        x = self.word_lstm_encoder(x)
        y = self.word_lstm_encoder(y)
        
        #compute a distance metric. This can be tricky and needs experimenting for different tasks.
        #Here we use two: a product between the embeddings and an euclidean distance
        #And do it separately for words and characters
        #You can also, in principle, just concatenate the outputs, but the branches together
        #but then you need to compute them twice (also feed y's through the x branch and x's through the y branch
        #to ensure the readout (classifier) comes out symmetrical and the sentences transitve
        
        #z = torch.cat([x*y, (x-y)**2, cx*cy, (cx-cy)**2], 1)
        z = torch.exp(-torch.sum(torch.abs(x - y), dim = 1))
        #finally output the probability that these two questions mean the same thing
        return z
        
