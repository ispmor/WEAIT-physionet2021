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
                 model_type='alpha',
                 classes = []):
        super(LSTM_ECG, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.sigmoid = nn.Sigmoid()
        self.when_bidirectional = 1 # if bidirectional = True, then it has to be equal to 2
        print(f'| LSTM_ECG')


        #self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # The linear layer that maps from hidden state space to tag space
        self.lstm_alpha1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        if model_type == 'alpha':
            self.lstm_alpha2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                                       num_layers=num_layers, batch_first=True, bidirectional=False)

            self.fc_1 = nn.Linear(hidden_size*541, 128)#hidden_size, 128)  # fully connected 1
            self.fc = nn.Linear(128, num_classes)  # fully connected last layer
        else:
            self.hidden_size=1
            self.num_layers=1
            self.input_size=1
            self.lstm_alpha1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                       num_layers=self.num_layers, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(input_size * 6 + 363 * 6 + 6, num_classes)

        self.relu = nn.ReLU()

    def forward(self, rr_x, rr_wavelets):
        if self.model_type == 'alpha':
            h_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # hidden state
            c_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # internal state
            h_1 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # hidden state
            c_1 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # internal state

            output_alpha1, (hn_alpha1, cn) = self.lstm_alpha1(rr_x, (h_0, c_0))  # lstm with input, hidden, and internal state
            output_alpha2, (hn_alpha2, cn) = self.lstm_alpha2(rr_wavelets, (h_1, c_1))  # lstm with input, hidden, and internal state
            tmp = torch.hstack((output_alpha1, output_alpha2))
            tmp = torch.flatten(tmp, start_dim=1)

            out = self.fc_1(tmp)  # first Dense
            out = self.relu(out)  # relu
            out = self.fc(out)  # Final Output
            return out
        else:
            h_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # hidden state
            c_0 = autograd.Variable(torch.zeros(self.num_layers * self.when_bidirectional, rr_x.size(0), self.hidden_size, device=torch.device('cuda:0')))  # internal state

            output_beta, (hn_beta, cn) = self.lstm_alpha1(rr_wavelets, (h_0, c_0))

            out = torch.squeeze(output_beta)
            out = self.relu(out)  # relu
            out = self.fc(out)  # Final Output
        return out



class BlendMLP(nn.Module):
    def __init__(self, modelA, modelB, num_classes):
        super(BlendMLP, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.linear = nn.Linear(2*num_classes, num_classes)

    def forward(self, rr_x, rr_wavelets, pca_features):
        x1 = self.modelA(rr_x, rr_wavelets)
        x2 = self.modelB(rr_x, pca_features)
        out = torch.cat((x1, x2), dim=1)
        out = self.linear(F.relu(out))
        return out