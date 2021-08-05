import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from datetime import date
import config
from scipy import stats
from scipy import signal
from torch import nn
from torch.nn import functional as F
#import matplotlib.pyplot as plt
from pywt import wavedec


today = date.today().strftime("%d%m%Y")
#def plot_scatter(*args, **kwargs):
    #plt.plot(*args, **kwargs)
    #plt.scatter(*args, **kwargs)
    
fs = 500.0
hum_freq = [60.0, 120.0, 240.0]
Q = 30.0

notch_60_b, notch_60_a = signal.iirnotch(hum_freq[0], Q, fs)
notch_120_b, notch_120_a = signal.iirnotch(hum_freq[1],Q, fs)
notch_240_b, notch_240_a = signal.iirnotch(hum_freq[2],Q, fs)

sos = signal.butter(10, 1, 'hp', fs=500, output='sos')


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


    
def get_data_with_labels(recording, forecast_length, backcast_length, batch_size, cuda, labels):
    x_train_batch, y = [], []
    if len(recording[0]) > 7500:
        recording = recording[:, 0:7500]
    
    if len(recording[0]) - backcast_length > backcast_length:
        for i in range(backcast_length, len(recording[0]) - backcast_length , forecast_length): # tutaj jest problem, w którym chce obniżyć ilość batchy wykorzystywanych z pliku, celem przyspieszeniea treningu
            x_train_batch.append(recording[:, i - backcast_length:i])
            y.append(labels)
    else:
        for i in range(4):
            x_train_batch.append(recording[:, 0:backcast_length])
            y.append(labels)
        
    x_train_batch = torch.tensor(x_train_batch, device=cuda, dtype=torch.float)  # [..., 0]
    y = torch.tensor(y, device=cuda,  dtype=torch.float)  # [..., 0]
    
  

    x_train, x_test, y_train, y_test = train_test_split(x_train_batch, y, test_size=0.2, train_size=0.8, random_state=17)
    data = data_generator(x_train, y_train, batch_size)

    return data,x_train, y_train, x_test, y_test
        

def evaluate_training(backcast_length, forecast_length, net, test_losses, x_test, y_test, the_lowest_error, device, experiment, plot_eval=False, step=0, file_name=""):
    net.eval()
    forecast = net(x_test.clone().detach())
    
    m = nn.BCEWithLogitsLoss()
    singular_loss = m(forecast, y_test).item()
    
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
    

def one_file_training_data(recording, single_peak_length, peaks):
    x = []
    peaks_len = len(peaks)
    prev_distance = 0
    next_distance = 0
    rr_features = []
    coeffs = []
    for i, peak in enumerate(peaks):
        if i == 0:
            prev_distance = peak
        else:
            prev_distance = peak - peaks[i-1]

        if i == peaks_len-1:
            continue
        else:
            next_distance = peaks[i+1] - peak

        if i < 5 and i < peaks_len - 5:
            avg = (sum(peaks[0:i]) + sum(peaks[i:i+5])) / float(i+5)
        elif 5 < i < peaks_len - 5:
            avg = sum(peaks[i-5: i+5]) / 10.0
        else:
            avg = (sum(peaks[i-5:i]) + sum(peaks[i:peaks_len-1])) / float(i+5)

        if peak < 125:
            signal = recording[:, 0: single_peak_length]
            a4, d4, d3, d2, d1 = wavedec(signal[:, ::2], 'db2', level=4)
            wavelet_features = np.hstack((a4, d4, d3, d2, d1))
        elif peak + 225 < len(recording[0]):
            signal = recording[:, peak - 125:peak + 225]
            a4, d4, d3, d2, d1 = wavedec(signal[:, ::2], 'db2', level=4)
            wavelet_features = np.hstack((a4, d4, d3, d2, d1))
        else:
            continue
        x.append(signal)
        rr_features.append([[prev_distance, next_distance, avg] for i in range(len(recording))])
        coeffs.append(wavelet_features)

    x = np.array(x, dtype=np.float)
    rr_features = np.array(rr_features, dtype=np.float)
    coeffs = np.asarray(coeffs,  dtype=np.float)

    return rr_features, x, coeffs



def equalize_signal_frequency(freq, recording_full):
    new_recording_full = []

    if freq == float(257):
        xp = [i * 1.9455 for i in range(recording_full.shape[1])]
        x = np.linspace(0, 30 * 60 * 500, 30 * 60 * 500)
        for lead in recording_full:
            new_lead = np.interp(x, xp, lead)
            new_recording_full.append(new_lead)
        new_recording_full = np.array(new_recording_full)

    if freq == float(1000):
        x_base = list(range(len(recording_full[0])))
        x_shortened = x_base[::2]
        new_recording_full = recording_full[:, ::2]

            
    #fig = plt.figure(1, figsize=(12, 10))
    #plt.grid()
    #plot_scatter(x_base[0:2000], recording_full[0][0:2000], color='b')
    #plot_scatter(x_shortened[0:1000], new_recording_full[0][0:1000], color='g')
    #plt.savefig("/home/puszkar/signal-500.png")
    #plt.close()
    
    return new_recording_full
            
    
    
def apply_notch_filters(input_signal):
    output_signal = signal.filtfilt(notch_60_b, notch_60_a, input_signal)
    output_signal = signal.filtfilt(notch_120_b, notch_120_a, output_signal)
    output_signal = signal.filtfilt(notch_240_b, notch_240_a, output_signal)
    output_signal = signal.sosfilt(sos, output_signal)
    
    return output_signal
    
    
    
    
    