import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd

import pdb


class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=1,
                 target_size=5,
                 input_size=10,
                 thetas_dims=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=17,
                 classes=[],
                 model_type='alpha'):
        super(NBeatsNet, self).__init__()
        self.classes = classes
        self.leads = []
        self.target_size = target_size
        self.input_size = input_size
        self.hidden_layer_units = hidden_layer_units
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack
        self.stack_types = stack_types
        self.stacks = []
        self.thetas_dim = thetas_dims
        self.parameters = []

        if model_type == 'alpha':
            linear_input_size = 353 * input_size
        else:
            self.linea_multiplier = input_size
            if input_size > 6:
                self.linea_multiplier = 6
            linear_input_size = input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier
        self.fc_linear = nn.Linear(353 * len(classes), len(classes))

        print(f'| N-Beats')

        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))
        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]
        print(f'| --  Stack {stack_type.title()} (#{stack_id}) (share_weights_in_stack={self.share_weights_in_stack})')
        blocks = []
        for block_id in range(self.nb_blocks_per_stack):
            block_init = NBeatsNet.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]  # pick up the last one when we share weights.
            else:
                block = block_init(self.hidden_layer_units, self.thetas_dim[stack_id], self.input_size,
                                   self.target_size, classes=len(self.classes))
                self.parameters.extend(block.parameters())
            print(f'     | -- {block}')
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        return GenericBlock

    def forward(self, backcast):
        forecast = torch.zeros(size=backcast.shape).cuda()
        for stack_id in range(len(self.stacks)):
            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](backcast)
                backcast = backcast - b
                forecast = forecast + f

        return backcast, forecast


def linspace(backcast_length, forecast_length):
    lin_space = np.linspace(-backcast_length, forecast_length, backcast_length + forecast_length)
    b_ls = lin_space[:backcast_length]
    f_ls = lin_space[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):
    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, share_thetas=False, classes=16):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.share_thetas = share_thetas
        self.fc1 = nn.Linear(backcast_length, units)
        self.fc2 = nn.Linear(units, units)
        self.fc3 = nn.Linear(units, units)
        self.fc4 = nn.Linear(units, units)
        self.backcast_linspace, self.forecast_linspace = linspace(backcast_length, forecast_length)
        self.classes = classes

        if share_thetas:
            self.theta_f_fc = self.theta_b_fc = nn.Linear(units, thetas_dim)
        else:
            self.theta_b_fc = nn.Linear(units, thetas_dim)
            self.theta_f_fc = nn.Linear(units, thetas_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \
               f'share_thetas={self.share_thetas}) at @{id(self)}'


class GenericBlock(Block):

    def __init__(self, units, thetas_dim, backcast_length=10, forecast_length=5, classes=16):
        super(GenericBlock, self).__init__(units, thetas_dim, backcast_length, forecast_length, classes=classes)

        self.backcast_fc = nn.Linear(thetas_dim, backcast_length)
        self.forecast_fc = nn.Linear(thetas_dim, backcast_length)  # forecast_length)

    def forward(self, x):
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))  # tutaj masz thetas_dim rozmiar

        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast


class Nbeats_alpha(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 classes=[],
                 model_type='alpha'):
        super(Nbeats_alpha, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.relu = nn.ReLU()

        self.nbeats_alpha1 = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                       nb_blocks_per_stack=2,
                                       target_size=num_classes,
                                       input_size=input_size,
                                       thetas_dims=(32, 32),
                                       classes=self.classes,
                                       hidden_layer_units=self.hidden_size)

        self.nbeats_alpha2 = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                       nb_blocks_per_stack=1,
                                       target_size=num_classes,
                                       input_size=input_size,
                                       thetas_dims=(32, 32),
                                       classes=self.classes,
                                       hidden_layer_units=hidden_size)

        self.fc_1 = nn.Linear(self.input_size * 541, 128)  # hidden_size, 128)  # fully connected 1
        self.fc = nn.Linear(128, num_classes)  # fully connected last layer

    def forward(self, rr_x, rr_wavelets):
        _, output_alpha1 = self.nbeats_alpha1(rr_x)  # lstm with input, hidden, and internal state
        _, output_alpha2 = self.nbeats_alpha2(rr_wavelets)  # lstm with input, hidden, and internal state

        tmp = torch.hstack((output_alpha1, output_alpha2))
        tmp = torch.flatten(tmp, start_dim=1)

        out = self.fc_1(tmp)  # first Dense
        out = self.relu(out)  # relu
        out = self.fc(out)  # Final Output
        return out


class Nbeats_beta(nn.Module):
    def __init__(self,
                 input_size,
                 num_classes,
                 hidden_size,
                 num_layers,
                 seq_length,
                 classes=[],
                 model_type='beta'):
        super(Nbeats_beta, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers  # number of layers
        self.hidden_size = hidden_size  # hidden state
        self.seq_length = seq_length  # sequence length
        self.model_type = model_type
        self.classes = classes
        self.relu = nn.ReLU()

        self.linea_multiplier = input_size
        if input_size > 6:
            self.linea_multiplier = 6
        self.hidden_size = 1
        self.num_layers = 1
        self.input_size = 1

        self.nbeats_beta = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK],
                                     nb_blocks_per_stack=1,
                                     target_size=num_classes,
                                     input_size=self.input_size,
                                     thetas_dims=(32, 32),
                                     classes=self.classes,
                                     hidden_layer_units=17)

        self.fc = nn.Linear(input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier,
                            num_classes)  # hidden_size, 128)  # fully connected 1# fully connected last layer

    def forward(self, pca_features):
        _, output_beta = self.nbeats_beta(pca_features)  # lstm with input, hidden, and internal state

        tmp = torch.squeeze(output_beta)
        out = self.relu(tmp)  # relu
        out = self.fc(out)  # Final Output
        return out






    
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
            self.linea_multiplier = input_size
            if input_size > 6:
                self.linea_multiplier = 6
            self.hidden_size=1
            self.num_layers=1
            self.input_size=1
            self.lstm_alpha1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                       num_layers=self.num_layers, batch_first=True, bidirectional=False)
            self.fc = nn.Linear(input_size * self.linea_multiplier + 363 * self.linea_multiplier + self.linea_multiplier, num_classes)

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
    def __init__(self, modelA, modelB, classes):
        super(BlendMLP, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classes = classes
        self.linear = nn.Linear(2*len(classes), len(classes))

    def forward(self, rr_x, rr_wavelets, pca_features):
        x1 = self.modelA(rr_x, rr_wavelets)
        #x2 = self.modelB(rr_x, pca_features) # FOR LSTM
        x2 = self.modelB(pca_features) #FOR NBEATS

        out = torch.cat((x1, x2), dim=1)
        out = self.linear(F.relu(out))
        return out