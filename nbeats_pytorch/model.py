import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd

import pdb


    
class LSTM_ECG(nn.Module):
    def __init__(self,
                 device,
                 window_size,
                 target_size,
                 hidden_dim,
                 classes = [],
                 leads = []):
        super(LSTM_ECG, self).__init__()
        self.classes = classes
        self.leads = leads
        print(f'| LSTM_ECG')

        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.device = device
        self.leads = leads
        self.lstm = nn.LSTM(window_size, hidden_dim, bidirectional=True)  
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2 * len(leads), target_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)
        self.to(device)
        
       
        
        
    def init_hidden(self):
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, len(self.leads), self.hidden_dim,device=self.device)),   
                autograd.Variable(torch.zeros(2, len(self.leads), self.hidden_dim,device=self.device))) 


    def forward(self, signal):
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(signal, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(signal), -1))
        tag_scores = self.softmax(tag_space)#F.log_softmax(tag_space, dim=1) #s
        print(tag_scores[-1])
        return tag_scores
