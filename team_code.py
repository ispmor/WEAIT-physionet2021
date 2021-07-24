#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib



####
from torch.utils.tensorboard import SummaryWriter
from nbeats_pytorch.model import LSTM_ECG
from torch.nn import functional as F
from torch import nn
from torch.utils import data as torch_data
import nbeats_additional_functions_2021 as naf
import os
import torch
from torch import optim
from config import exp_net_params as exp
import h5py
from h5class import HDF5Dataset
import json

# import matplotlib.pyplot as plt


twelve_lead_model_filename = '12_lead_model.th'
six_lead_model_filename = '6_lead_model.th'
four_lead_model_filename = '4_lead_model.th'
three_lead_model_filename = '3_lead_model.th'
two_lead_model_filename = '2_lead_model.th'

twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
leads_set = set([twelve_leads, six_leads, four_leads, three_leads, two_leads]) #USUNIĘTE DŁUŻSZE TRENOWANIE MODELI

single_peak_length = exp["single_peak_length"]
forecast_length = exp["forecast_length"]
batch_size = exp["batch_size"]
backcast_length = exp["backcast_length"]
hidden = exp["hidden_layer_units"]
nb_blocks_per_stack = exp["nb_blocks_per_stack"]
thetas_dim = exp["thetas_dim"]
window_size = exp["window_size"]


#cuda0 = torch.cuda.set_device(0)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.pin_memory=False
            

classes_numbers = dict()
class_files_numbers = dict()


################################################################################
#
# Training function
#
################################################################################

# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()

    for header_file in header_files:
        header = load_header(header_file)
        classes_from_header = get_labels(header)
        classes |= set(classes_from_header)
        for c in classes_from_header:
            if c in class_files_numbers:
                class_files_numbers[c] += 1
            else:
                class_files_numbers[c] = 1

    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    print(classes)
    class_index = dict()
    for i, c in enumerate(classes):
        class_index[c] = i



    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    for leads in leads_set:

        leads_idx = np.array(list(range(len(leads))))

        name = get_model_filename(leads)
        filename = os.path.join(model_directory, name)
        experiment = SummaryWriter()

    

        torch.manual_seed(17)
        init_dataset = list(range(num_recordings))
        lengths = [int(len(init_dataset) * 0.8), len(init_dataset) - int(len(init_dataset) * 0.8)]
        data_training, data_validation = torch_data.random_split(init_dataset, lengths)
        weights = None
        ################ CREATE HDF5 DATABASE #############################3
        if not os.path.isfile('cinc_database_training.h5'): #_{len(leads)}_training.h5'):
            create_hdf5_db(data_training, num_classes, header_files, recording_files, classes, twelve_leads, isTraining=True)
            global classes_numbers
            summed_classes = sum(classes_numbers.values())
            sorted_classes_numbers = dict(sorted(classes_numbers.items(), key=lambda x: int(x[0])))
            weights = torch.tensor([c / summed_classes for c in sorted_classes_numbers.values()], device=device)

            np.savetxt("weights_training.csv", weights.detach().cpu().numpy(), delimiter=',')
        if not os.path.isfile('cinc_database_validation.h5'): #{len(leads)}_validation.h5'):
            create_hdf5_db(data_validation, num_classes, header_files, recording_files, classes, twelve_leads,
                           isTraining=False)
        if weights is None and os.path.isfile('cinc_database_training.h5'):
            weights = torch.tensor(np.loadtxt('weights_training.csv', delimiter=','), device=device)
        if len(classes_numbers.values()) == 0 and os.path.isfile("classes_in_h5_occurrences.json"):
            with open("classes_in_h5_occurrences.json", 'r') as f:
                classes_numbers = json.load(f)
        elif len(classes_numbers.values()) != 0 and not os.path.isfile("classes_in_h5_occurrences.json"):
            with open("classes_in_h5_occurrences.json", 'w') as f:
                json.dump(classes_numbers, f)

        classes_over_50000 = dict()
        index_mapping_from_normal_to_over_5000 = dict()
        tmp_iterator = 0
        for c in classes:
            if classes_numbers[c] > 50000:
                classes_over_50000[c] = tmp_iterator
                index_mapping_from_normal_to_over_5000[class_index[c]]=tmp_iterator
                tmp_iterator +=1
        summed_classes = sum([classes_numbers[key] for key in classes_over_50000.keys()])
        sorted_classes_numbers = dict(sorted([(k, classes_numbers[k]) for k in classes_over_50000.keys()], key=lambda x: int(x[0])))
        weights = torch.tensor([c / summed_classes for c in sorted_classes_numbers.values()], device=device)

        print("Creating LSTM")
        net = LSTM_ECG(device, forecast_length, len(classes_over_50000.keys()), hidden_dim=1256, classes=classes_over_50000.keys(), leads=leads)
        net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

        training_dataset = HDF5Dataset('./' + 'cinc_database_training.h5', recursive=False,
                                       load_data=False,
                                       data_cache_size=4, transform=None, leads=leads_idx)
        validation_dataset = HDF5Dataset('./' + 'cinc_database_validation.h5', recursive=False,
                                         load_data=False,
                                         data_cache_size=4, transform=None, leads=leads_idx)
        print("Przed trainnig data loaderem")
        training_data_loader = torch_data.DataLoader(training_dataset, batch_size=5000, shuffle=True, num_workers=6)
        print("Przed data loader walidacyjnym ")
        validation_data_loader = torch_data.DataLoader(validation_dataset, batch_size=5000, shuffle=True, num_workers=6)

        print("data_loader", training_data_loader)

        n_epochs_stop = 6
        epochs_no_improve = 0
        early_stop = False
        min_val_loss = 999

        print("Przed epokami")

        num_epochs = 20
        m = nn.BCEWithLogitsLoss(pos_weight=weights)
        for epoch in range(num_epochs):
            local_step = 0
            epoch_loss = []

            for x, y in training_data_loader:
                local_step += 1
                print("Batch number:", local_step)
                net.train()
                forecast = net(x.to(device))  # .to(device)) #Dodaje od
                print(forecast[0])
                y_selected = np.zeros(shape=(y.shape[0], len(classes_over_50000.keys())))

                for i, vector in enumerate(y):
                    indexes = (vector == 1).nonzero().tolist()
                    if len(indexes) > 0:
                        indexes = indexes[0]
                    for normal_index in indexes:
                        if normal_index in index_mapping_from_normal_to_over_5000:
                            y_selected[i][index_mapping_from_normal_to_over_5000[normal_index]] = 1.0

                y_selected = torch.tensor(y_selected)
                y_cuda = y_selected.to(device)
                loss = m(forecast, y_cuda)  # torch.zeros(size=(16,)))
                print(y_selected[0])
                epoch_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean = torch.mean(torch.stack(epoch_loss))
            print("Epoch: %d Training loss: %f" % (epoch, mean))
            name_exp = 'train_loss_' + str(len(leads)) + "LSTM"

            with torch.no_grad():
                epoch_loss = []
                net.eval()
                print("Net in eval mode")
                for x, y in validation_data_loader:
                    print("Step in validation loop")
                    forecast = net(x.to(device))  # .to(device))\
                    y_selected = np.zeros(shape=(y.shape[0], len(classes_over_50000.keys())))

                    for i, vector in enumerate(y):
                        indexes = (vector == 1).nonzero().tolist()
                        if len(indexes) > 0:
                            indexes = indexes[0]
                        for normal_index in indexes:
                            if normal_index in index_mapping_from_normal_to_over_5000:
                                y_selected[i][index_mapping_from_normal_to_over_5000[normal_index]] = 1.0

                    y_selected = torch.tensor(y_selected)
                    y_cuda = y_selected.to(device)
                    loss = m(forecast, y_cuda)
                    epoch_loss.append(loss)

                mean_val = torch.mean(torch.stack(epoch_loss))
                print("Epoch: %d Validation loss: %f" % (epoch, mean_val))

                experiment.add_scalars(name_exp, {
                    'BCEWithLogitsLoss': mean,
                 #   'MSELoss': mean_mse,
                    'ValidationBCEWithLogitsLoss': mean_val,
                }, epoch)

                if mean_val < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = mean_val
                    print(f'Savining {len(leads)}-lead ECG model, epoch: {epoch}...')
                    save(filename, net, optimizer, list(sorted_classes_numbers.keys()), leads)
                else:
                    epochs_no_improve += 1

                if epoch > 7 and epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
                    break
                if torch.isnan(mean_val).any():
                    print("NaN detected, stopping")
                    break

            scheduler.step()


################################################################################
#
# File I/O functions
#
################################################################################
# create HDF5 datase
def create_hdf5_db(num_recordings, num_classes, header_files, recording_files, classes, leads, isTraining=True):
    group = None
    if isTraining:
        group = 'training'
    else:
        group = 'validation'

    with h5py.File(f'cinc_database_{group}.h5', 'w') as h5file:

        grp = h5file.create_group(group)

        dset = grp.create_dataset("data", (1, len(leads), window_size),
                                  maxshape=(None, len(leads), window_size), dtype='f',
                                  chunks=(1, len(leads), window_size))
        lset = grp.create_dataset("label", (1, num_classes), maxshape=(None, num_classes), dtype='f',
                                  chunks=(1, num_classes))
        counter = 0
        for i in num_recordings:
            print('    {}/{}...'.format(counter + 1, len(num_recordings)))
            counter += 1
            # Load header and recording.
            header = load_header(header_files[i])
            classes_from_header = get_labels(header)

            recording = np.array(load_recording(recording_files[i]), dtype=np.float32)

            recording_full = get_leads_values(header, recording, leads)
            current_labels = get_labels(header)
            freq = get_frequency(header)
            if freq != float(500):
                recording_full = naf.equalize_signal_frequency(freq, recording_full)

            recording_full = naf.one_file_training_data(recording_full, window_size, device)
            local_label = np.zeros((num_classes,), dtype=np.bool)
            for label in current_labels:
                if label in classes:
                    j = classes.index(label)
                    local_label[j] = True

            new_windows = recording_full.shape[0]
            dset.resize(dset.shape[0] + new_windows, axis=0)
            dset[-new_windows:] = recording_full

            label_pack = [local_label for i in range(recording_full.shape[0])]
            lset.resize(lset.shape[0] + new_windows, axis=0)
            lset[-new_windows:] = label_pack

            global classes_numbers
            for c in classes_from_header:
                for i in range(new_windows):
                    if c in classes_numbers:
                        classes_numbers[c] += 1
                    else:
                        classes_numbers[c] = 1

    print(f'Successfully created {group} dataset')


# Save your trained models.
def save(checkpoint_name, model, optimiser, classes, leads):
    torch.save({
        'classes': classes,
        'leads': leads,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict()
    }, checkpoint_name)


# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.

# Generic function for loading a model.
def load_model(model_directory, leads):
    filename = os.path.join(model_directory, get_model_filename(leads))
    checkpoint = torch.load(filename, map_location=torch.device('cuda:0'))

    model = LSTM_ECG(device, forecast_length, len(checkpoint["classes"]), hidden_dim=1256, classes=checkpoint["classes"], leads=leads)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.leads = checkpoint['leads']
    model.cuda()
    print(f'Restored checkpoint from {filename}.')
    return model


# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model.classes
    leads = model.leads

    features = get_leads_values(header, recording.astype(np.float), leads)

    features = torch.Tensor(naf.one_file_training_data(features, window_size, device))
    # Predict labels and probabilities.
    probabilities = model(features.to(device))
 
    probabilities_mean = torch.mean(probabilities, 0).detach().cpu().numpy()

    labels = probabilities_mean.copy()
    labels[labels != labels.max()] = 0
    labels[labels != 0] = 1

    # probabilities = classifier.predict_proba(features)
    # probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities_mean


# Define the filename(s) for the trained models. This function is not required. You can change or remove it.
def get_model_filename(leads):
    number = len(leads)
    if number == 12:
        return twelve_lead_model_filename
    elif number == 6:
        return six_lead_model_filename
    elif number == 4:
        return four_lead_model_filename
    elif number == 3:
        return three_lead_model_filename
    else:
        return two_lead_model_filename


################################################################################
#
# Feature extraction function
#
################################################################################
# Extract features from the header and recording.
def get_features(header, recording, leads):
    # Extract age.
    age = get_age(header)
    if age is None:
        age = float('nan')

    # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
    sex = get_sex(header)
    if sex in ('Female', 'female', 'F', 'f'):
        sex = 0
    elif sex in ('Male', 'male', 'M', 'm'):
        sex = 1
    else:
        sex = float('nan')

    # Reorder/reselect leads in recordings.
    recording = choose_leads(recording, header, leads)

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x ** 2) / np.size(x))

    return age, sex, rms


def get_leads_values(header, recording, leads):
    # Reorder/reselect leads in recordings.
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adc_gains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    return recording



