import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd

import pdb


    
class LSTM_ECG(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 classes = []):
        super(LSTM_ECG, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length

        self.classes = classes
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1 # if bidirectional = True, then it has to be equal to 2
        print(f'| LSTM_ECG')


        #self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstm_alpha1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        self.lstm_alpha2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True, bidirectional=False)
       # self.hidden2tag = nn.Linear(hidden_size * num_layers * input_size, num_classes)

        self.fc_1 = nn.Linear(hidden_size*541, 128)#hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

        #self.fc_alpha = nn.Linear(hidden_size*541, num_classes)

        self.relu = nn.ReLU()

    #   def init_hidden(self):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
 #       return (autograd.Variable(torch.zeros(2, self.input_size, self.hidden_size,device=self.device)),
  #              autograd.Variable(torch.zeros(2, self.input_size, self.hidden_size,device=self.device)))

    def forward(self, rr_x, rr_wavelets):
        h_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # hidden state
        c_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # internal state
        h_1 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # hidden state
        c_1 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # internal state

        #ALPHA1
        output_alpha1, (hn_alpha1, cn) = self.lstm_alpha1(rr_x, (h_0, c_0))  # lstm with input, hidden, and internal state
        hn_alpha1 = hn_alpha1[self.num_layers-1].view(-1, self.hidden_size)  # reshaping the data for Dense layer next
        #ALPHA2
        output_alpha2, (hn_alpha2, cn) = self.lstm_alpha2(rr_wavelets, (h_1, c_1))  # lstm with input, hidden, and internal state
        hn_alpha2 = hn_alpha2[self.num_layers - 1].view(-1, self.hidden_size)  # reshaping the data for Dense layer next

        tmp = torch.hstack((output_alpha1, output_alpha2))
        tmp = torch.flatten(tmp, start_dim=1)
        #out = self.relu(tmp)
        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out