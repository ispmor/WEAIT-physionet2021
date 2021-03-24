import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from datetime import date
import config
from scipy import stats
from torch import nn
from torch.nn import functional as F




today = date.today().strftime("%d%m%Y")
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


def data_generator(x, y, batch_size):
    while True:
        for xy_pair in split((x, y), batch_size):
            yield xy_pair


def split(arr, size):
    arrays = []
    while len(arr) > size:
        slice_ = arr[:size]
        arrays.append(slice_)
        arr = arr[size:]
    arrays.append(arr)
    return arrays


def batcher(dataset, batch_size, infinite=False):
    while True:
        x, y = dataset
        for x_, y_ in zip(split(x, batch_size), split(y, batch_size)):
            yield x_, y_
        if not infinite:
            break


def load(checkpoint_name, model, optimiser):
    if os.path.exists(checkpoint_name):
        checkpoint = torch.load(checkpoint_name, map_location=torch.device('cuda:0'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.cuda()
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        grad_step = checkpoint['grad_step']
        #print(f'Restored checkpoint from {checkpoint_name}.')
        return grad_step
    return 0


def save(checkpoint_name, model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict()
    }, checkpoint_name)

        

def evaluate_training(backcast_length, forecast_length, net, test_losses, x_test, y_test, the_lowest_error, device, experiment, plot_eval=False, step=0, file_name=""):
    net.eval()
    _, forecast = net(x_test.clone().detach())
    
    m = nn.BCEWithLogitsLoss()
    singular_loss = m(forecast, torch.zeros(size=(16,))).item()
    
    #singular_loss = F.mse_loss(forecast, y_test.clone().detach()).item()
        
    test_losses.append(singular_loss)
    if singular_loss < the_lowest_error[-1]:
            the_lowest_error.append(singular_loss)           
           
    
    #if plot_eval:
    #    p = forecast.detach().cpu().numpy()
    #    fig = plt.figure(1, figsize=(12, 10))
    #    for _, i in enumerate(np.random.choice(range(len(x_test)), size=1, replace=False)):
    #        ff, xx, yy = p[i], x_test.detach().cpu().numpy()[i], y_test.detach().cpu().numpy()[i]
    #        for plot_id in range(1,10): 
    #            plt.subplot(340 + plot_id) ########### tu były zmiany
    #            plt.grid()
    #            plt.ylabel("Values normalised")
    #            plt.xlabel("Time (1/500 s)")
    #            plt.title(file_name + "Lead: " + str(plot_id))
    #            plot_scatter(range(0, backcast_length), xx[plot_id], color='b')
    #            plot_scatter(range(backcast_length, backcast_length + forecast_length), yy[plot_id], color='g')
    #            plot_scatter(range(backcast_length, backcast_length + forecast_length), ff[plot_id], color='r')
    #    experiment.log_image('epoch_test_eval_visualisation', fig)
    #    if not os.path.exists(f"/home/puszkar/ecg/results/images/training_eval/{today}"):
    #        os.mkdir(f"/home/puszkar/ecg/results/images/training_eval/{today}")
    #    plt.savefig(f"/home/puszkar/ecg/results/images/training_eval/{today}/latest_eval.png")
    #   plt.close()
    
    return singular_loss
    

def one_file_training_data(recording, forecast_length, backcast_length, batch_size, cuda, lead=3):
    x_train_batch, y = [], []
    if len(recording[0]) > 7500:
        recording = recording[:, 0:7500]
    
    for i in range(backcast_length, len(recording[0]) - forecast_length):
        x_train_batch.append(recording[:, i - backcast_length:i])
        y.append(recording[:, i:i + forecast_length])

    x_train_batch = torch.tensor(x_train_batch, device=cuda, dtype=torch.float)  # [..., 0]
    y = torch.tensor(y, device=cuda,  dtype=torch.float)  # [..., 0]
    

    x_train, x_test, y_train, y_test = train_test_split(x_train_batch, y, test_size=0.2, random_state=17)
    data = data_generator(x_train, y_train, batch_size)

    return data,x_train, y_train, x_test, y_test


