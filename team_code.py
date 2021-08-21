#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np

####
from torch.utils.tensorboard import SummaryWriter

from nbeats_pytorch.model import LSTM_ECG, BlendMLP, Nbeats_alpha, Nbeats_beta
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
from scipy.signal import butter, filtfilt, lfilter

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
leads_set = [twelve_leads, six_leads, four_leads, three_leads, two_leads]  # USUNIĘTE DŁUŻSZE TRENOWANIE MODELI

single_peak_length = exp["single_peak_length"]
forecast_length = exp["forecast_length"]
batch_size = exp["batch_size"]
backcast_length = exp["backcast_length"]
hidden = 17
nb_blocks_per_stack = exp["nb_blocks_per_stack"]
thetas_dim = exp["thetas_dim"]
window_size = exp["window_size"]  # rr features

torch.cuda.set_device(0)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.pin_memory = False

classes_numbers = dict()
class_files_numbers = dict()

sigmoid = nn.Sigmoid()


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
        classes = sorted(classes)  # Sort classes alphanumerically otherwise.
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

        global classes_numbers



        selected_classes = ['6374002', '10370003', '17338001', '39732003', '47665007', '59118001', '59931005',
                            '111975006', '164889003', '164890007', '164909002', '164917005', '164934002',
                            '164947007', '251146004', '270492004', '284470004', '365413008', '426177001', '426627000',
                            '426783006', '427084000', '427393009', '445118002', '698252002', '713426002']
        #, '427172004','63593006',      '713427006'   , '733534002']
        num_classes = len(selected_classes)

        print("SELECTED CLASSES: ", selected_classes)
        weights = None
        ################ CREATE HDF5 DATABASE #############################3
        if not os.path.isfile('cinc_database_training.h5'):  # _{len(leads)}_training.h5'):
            create_hdf5_db(data_training, num_classes, header_files, recording_files, selected_classes, twelve_leads,
                           isTraining=1, selected_classes=selected_classes)

            summed_classes = sum(classes_numbers.values())

            sorted_classes_numbers = dict(sorted(classes_numbers.items(), key=lambda x: int(x[0])))

            weights = calculate_pos_weights(sorted_classes_numbers.values())
            np.savetxt("weights_training.csv", weights.detach().cpu().numpy(), delimiter=',')


        if not os.path.isfile('cinc_database_validation.h5'):  # {len(leads)}_validation.h5'):
            create_hdf5_db(data_validation, num_classes, header_files, recording_files, selected_classes, twelve_leads,
                           isTraining=0, selected_classes=selected_classes)
        if weights is None and os.path.isfile('cinc_database_training.h5'):
            weights = torch.tensor(np.loadtxt('weights_training.csv', delimiter=','), device=device)

        if len(classes_numbers.values()) == 0 and os.path.isfile("classes_in_h5_occurrences_new.json"):
            with open("classes_in_h5_occurrences_new.json", 'r') as f:
                classes_numbers = json.load(f)
        elif len(classes_numbers.values()) != 0 and not os.path.isfile("classes_in_h5_occurrences_new.json"):
            with open("classes_in_h5_occurrences_new.json", 'w') as f:
                json.dump(classes_numbers, f)

        classes_to_classify = dict()
        index_mapping_from_normal_to_selected = dict()
        tmp_iterator = 0
        for c in classes:
            if c in selected_classes:
                classes_to_classify[c] = tmp_iterator
                index_mapping_from_normal_to_selected[class_index[c]] = tmp_iterator
                tmp_iterator += 1
        summed_classes = sum([classes_numbers[key] for key in selected_classes])
        sorted_classes_numbers = dict(
            sorted([(k, classes_numbers[k]) for k in classes_to_classify.keys()], key=lambda x: int(x[0])))

        weights = calculate_pos_weights(sorted_classes_numbers.values())

        print("Creating NBEATS")
        net = Nbeats_alpha(input_size=len(leads),
                           num_classes=len(selected_classes),
                           hidden_size=17,
                           seq_length=353,
                           model_type='alpha',
                           classes=selected_classes,
                           num_layers=1)

        net.cuda()

        net_beta = Nbeats_beta(input_size=len(leads),
                               num_classes=len(selected_classes),
                               hidden_size=1,
                               seq_length=353,
                               model_type='beta',
                               classes=selected_classes,
                               num_layers=1)
        net_beta.cuda()

        model = BlendMLP(net, net_beta, selected_classes)
        model.cuda()

        training_dataset = HDF5Dataset('./' + 'cinc_database_training.h5', recursive=False,
                                       load_data=False,
                                       data_cache_size=4, transform=None, leads=leads_idx)
        validation_dataset = HDF5Dataset('./' + 'cinc_database_validation.h5', recursive=False,
                                         load_data=False,
                                         data_cache_size=4, transform=None, leads=leads_idx)

        print("Przed trainnig data loaderem")
        training_data_loader = torch_data.DataLoader(training_dataset, batch_size=2500, shuffle=True, num_workers=6)
        print("Przed data loader walidacyjnym ")
        validation_data_loader = torch_data.DataLoader(validation_dataset, batch_size=2500, shuffle=True, num_workers=6)

        print("data_loader", training_data_loader)

        n_epochs_stop = 6
        epochs_no_improve = 0
        early_stop = False
        min_val_loss = 999

        print("Przed epokami")

        num_epochs = 25


        criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        name_exp = 'train_loss_' + str(len(leads)) + "LSTM" + " scheduler:Step"
        for epoch in range(num_epochs):
            local_step = 0
            epoch_loss = []

            for x, y, rr_features, wavelet_features in training_data_loader:
                x = torch.transpose(x, 1, 2)
                rr_features = torch.transpose(rr_features, 1, 2)
                wavelet_features = torch.transpose(wavelet_features, 1, 2)

                rr_x = torch.hstack((rr_features, x))
                rr_wavelets = torch.hstack((rr_features, wavelet_features))

                pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
                pca_features = torch.pca_lowrank(pre_pca)
                pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                             pca_features[2].reshape(pca_features[2].shape[0], -1)))
                pca_features = pca_features[:, :, None]

                local_step += 1
                print("Batch number:", local_step)
                model.train()

                forecast = model(rr_x.to(device), rr_wavelets.to(device), pca_features.to(device))

                print(forecast[0])


                y_selected = torch.tensor(y, device=device)
                loss = criterion(forecast, y_selected)  # torch.zeros(size=(16,)))
                print(y_selected[0])
                epoch_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean = torch.mean(torch.stack(epoch_loss))
            print("Epoch: %d Training loss: %f" % (epoch, mean))

            with torch.no_grad():
                epoch_loss1 = []

                model.eval()
                print("Net in eval mode")
                for x, y, rr_features, wavelet_features in validation_data_loader:
                    x = torch.transpose(x, 1, 2)
                    rr_features = torch.transpose(rr_features, 1, 2)
                    wavelet_features = torch.transpose(wavelet_features, 1, 2)

                    rr_x = torch.hstack((rr_features, x))
                    rr_wavelets = torch.hstack((rr_features, wavelet_features))

                    pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
                    pca_features = torch.pca_lowrank(pre_pca)
                    pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                                 pca_features[2].reshape(pca_features[2].shape[0], -1)))
                    pca_features = pca_features[:, :, None]

                    print("Step in validation loop")
                    # forecast = net(rr_x.to(device), rr_wavelets.to(device))  # .to(device))\
                    forecast = model(rr_x.to(device), rr_wavelets.to(device),
                                     pca_features.to(device))  # , rr_wavelets.to(device), pca_features.to(device))


                    y_selected = torch.tensor(y, device=device) # <- zmienione
                    loss = criterion(forecast, y_selected)
                    epoch_loss1.append(loss)

                mean_val1 = torch.mean(torch.stack(epoch_loss1))
                print("Epoch: %d Validation loss: %f" % (epoch, mean_val1))

                experiment.add_scalars(name_exp, {
                    'BCEWithLogitsLoss': mean,
                    'ValidationBCEWithLogitsLoss-only_14_classes': mean_val1,
                }, epoch)
                print("not improving since:", epochs_no_improve)

                if mean_val1 < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = mean_val1
                    print(f'Savining {len(leads)}-lead ECG model, epoch: {epoch}...')
                    save(filename, model, optimizer, list(sorted_classes_numbers.keys()), leads)
                else:
                    epochs_no_improve += 1

                if epoch > 10 and epochs_no_improve >= n_epochs_stop:
                    print('Early stopping!')
                    early_stop = True
                    break
                if torch.isnan(mean_val1).any():
                    print("NaN detected, stopping")
                    break


################################################################################
#
# File I/O functions
#
################################################################################
# create HDF5 datase

def calculate_pos_weights(class_counts):
    all_counts = sum(class_counts)
    neg_counts = [all_counts-pos_count for pos_count in class_counts]
    pos_weights = [neg_count / (pos_count + 1e-5) for (pos_count, neg_count) in  zip(class_counts,  neg_counts)]
    return torch.as_tensor(pos_weights, dtype=torch.float, device=device)

def create_hdf5_db(num_recordings, num_classes, header_files, recording_files, classes, leads, isTraining=1,
                   selected_classes=[]):
    group = None
    if isTraining == 1:
        group = 'training'
    elif isTraining == 0:
        group = 'validation'
    else:
        group = 'validation2'

    with h5py.File(f'cinc_database_{group}.h5', 'w') as h5file:

        grp = h5file.create_group(group)

        dset = grp.create_dataset("data", (1, len(leads), window_size),
                                  maxshape=(None, len(leads), window_size), dtype='f',
                                  chunks=(1, len(leads), window_size))
        lset = grp.create_dataset("label", (1, num_classes), maxshape=(None, num_classes), dtype='f',
                                  chunks=(1, num_classes))
        rrset = grp.create_dataset("rr_features", (1, len(leads), 3), maxshape=(None, len(leads), 3), dtype='f',
                                   chunks=(1, len(leads), 3))
        waveset = grp.create_dataset("wavelet_features", (1, len(leads), 185), maxshape=(None, len(leads), 185),
                                     dtype='f',
                                     chunks=(1, len(leads), 185))

        counter = 0
        for i in num_recordings:
            print('    {}/{}...'.format(counter + 1, len(num_recordings)))
            counter += 1
            # Load header and recording.
            header = load_header(header_files[i])
            classes_from_header = get_labels(header)
            if '733534002' in classes_from_header:
                classes_from_header[classes_from_header.index('733534002')] = '164909002'
                classes_from_header = list(set(classes_from_header))
            if '713427006' in classes_from_header:
                classes_from_header[classes_from_header.index('713427006')] = '59118001'
                classes_from_header = list(set(classes_from_header))
            if '63593006' in classes_from_header:
                classes_from_header[classes_from_header.index('63593006')] = '284470004'
                classes_from_header = list(set(classes_from_header))
            if '427172004' in classes_from_header:
                classes_from_header[classes_from_header.index('427172004')] = '17338001'
                classes_from_header = list(set(classes_from_header))

            class_in_file = False
            if isTraining < 2:
                s1 = set(classes_from_header)
                s2 = set(selected_classes)
                if not s1.intersection(s2):
                    continue
                print("S1: ", s1, "\nS2: ", s2, "\nCommon: ", s1.intersection(s2))
            recording = np.array(load_recording(recording_files[i]), dtype=np.float32)

            recording_full = get_leads_values(header, recording, leads)
            current_labels = get_labels(header)
            freq = get_frequency(header)
            if freq != float(500):
                recording_full = naf.equalize_signal_frequency(freq, recording_full)

            if recording_full.max() == 0 and recording_full.min() == 0:
                continue
            peaks = pan_tompkins_detector(500, recording_full[0])

            rr_features, recording_full, wavelet_features = naf.one_file_training_data(recording_full, window_size,
                                                                                       peaks)

            local_label = np.zeros((num_classes,), dtype=np.bool)
            for label in current_labels:
                if label in classes:
                    j = classes.index(label)
                    local_label[j] = True

            new_windows = recording_full.shape[0]
            if new_windows == 0:
                continue
            dset.resize(dset.shape[0] + new_windows, axis=0)
            dset[-new_windows:] = recording_full

            label_pack = [local_label for i in range(recording_full.shape[0])]
            lset.resize(lset.shape[0] + new_windows, axis=0)
            lset[-new_windows:] = label_pack

            rrset.resize(rrset.shape[0] + new_windows, axis=0)
            rrset[-new_windows:] = rr_features

            waveset.resize(waveset.shape[0] + new_windows, axis=0)
            if wavelet_features.shape[0] != new_windows:
                waveset[-new_windows:] = wavelet_features[:-1]
            else:
                waveset[-new_windows:] = wavelet_features

            global classes_numbers
            for c in classes_from_header:
                if c in selected_classes:
                    for i in range(new_windows):  #
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
    torch.cuda.set_device(0)

    filename = os.path.join(model_directory, get_model_filename(leads))
    checkpoint = torch.load(filename, map_location=torch.device('cuda:0'))

    # model = LSTM_ECG(device, single_peak_length, len(checkpoint["classes"]), hidden_dim=1256, classes=checkpoint["classes"], leads=leads)

    net = Nbeats_alpha(input_size=len(leads),
                       num_classes=len(checkpoint['classes']),
                       hidden_size=17,
                       seq_length=353,
                       model_type='alpha',
                       classes=checkpoint['classes'],
                       num_layers=1)

    net_beta = Nbeats_beta(input_size=len(leads),
                           num_classes=len(checkpoint['classes']),
                           hidden_size=1,
                           seq_length=353,
                           model_type='beta',
                           classes=checkpoint['classes'],
                           num_layers=1)

    model = BlendMLP(net, net_beta, checkpoint["classes"])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.leads = checkpoint['leads']
    model.cuda()
    print(f'Restored checkpoint from {filename}.')

    return model


# Generic function for running a trained model.
def run_model(model, header, recording):
    global sigmoid
    classes = model.classes
    leads = model.leads

    x_features = get_leads_values(header, recording.astype(np.float), leads)
    freq = get_frequency(header)
    if freq != float(500):
        x_features = naf.equalize_signal_frequency(freq, x_features)

    peaks = pan_tompkins_detector(500, x_features[0])

    rr_features, x_features, wavelet_features = naf.one_file_training_data(x_features, window_size, peaks)
    x_features = torch.Tensor(x_features)
    rr_features = torch.Tensor(rr_features)
    wavelet_features = torch.Tensor(wavelet_features)

    # Predict labels and probabilities.
    if len(x_features) == 0:
        labels = np.zeros(len(classes))
        probabilities_mean = np.zeros(len(classes))
        return classes, labels, probabilities_mean
    else:
        x = torch.transpose(x_features, 1, 2)
        rr_features = torch.transpose(rr_features, 1, 2)
        wavelet_features = torch.transpose(wavelet_features, 1, 2)

        rr_x = torch.hstack((rr_features, x))
        rr_wavelets = torch.hstack((rr_features, wavelet_features))

        pre_pca = torch.hstack((rr_features, x[:, ::2, :], wavelet_features))
        pca_features = torch.pca_lowrank(pre_pca)
        pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1],
                                     pca_features[2].reshape(pca_features[2].shape[0], -1)))
        pca_features = pca_features[:, :, None]

        with torch.no_grad():
            scores = model(rr_x.to(device), rr_wavelets.to(device), pca_features.to(device))
            del rr_x, rr_wavelets, rr_features, x, pca_features, pre_pca
            probabilities = nn.functional.sigmoid(scores)
            probabilities_mean = torch.mean(probabilities, 0).detach().cpu().numpy()
          # labels = np.zeros(len(probabilities_mean), type=np.bool)
            thresholds_per_class = [0.92912185, 0.99383825, 0.9807903,  0.8201744,  0.84914022, 0.80809229,0.7739887,  0.983991,   0.7987251,  0.9289746,  0.98765594, 0.85153157,0.9495117,  0.76390505, 0.72608936, 0.84350013, 0.91662383, 0.9896282, 0.90384232, 0.8161232, 0.8286067, 0.7592844, 0.67744523, 0.66565263,0.9370738 ,0.7162824 ]
            labels = probabilities_mean > 0.625
            #for i, thr in enumerate(thresholds_per_class):
            #    if probabilities_mean[i] > thr:
            #        labels[i] = 1
            #    else:
            #        labels[i] = 0

            print(labels)
            print(probabilities_mean)

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


def pan_tompkins_detector(fs, unfiltered_ecg, MWA_name='cumulative'):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm.
    In: IEEE Transactions on Biomedical Engineering
    BME-32.3 (1985), pp. 230–236.
    """

    f1 = 5 / fs
    f2 = 15 / fs

    b, a = butter(1, [f1 * 2, f2 * 2], btype='bandpass')

    filtered_ecg = lfilter(b, a, unfiltered_ecg)

    diff = np.diff(filtered_ecg)

    squared = diff * diff

    N = int(0.12 * fs)
    mwa = MWA_from_name(MWA_name)(squared, N)
    mwa[:int(0.2 * fs)] = 0

    mwa_peaks = panPeakDetect(mwa, fs)

    return mwa_peaks


def MWA_from_name(function_name):
    if function_name == "cumulative":
        return MWA_cumulative
    elif function_name == "convolve":
        return MWA_convolve
    elif function_name == "original":
        return MWA_original
    else:
        raise RuntimeError('invalid moving average function!')


# Fast implementation of moving window average with numpy's cumsum function
def MWA_cumulative(input_array, window_size):
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


# Original Function
def MWA_original(input_array, window_size):
    mwa = np.zeros(len(input_array))
    mwa[0] = input_array[0]

    for i in range(2, len(input_array) + 1):
        if i < window_size:
            section = input_array[0:i]
        else:
            section = input_array[i - window_size:i]

        mwa[i - 1] = np.mean(section)

    return mwa


# Fast moving window average implemented with 1D convolution
def MWA_convolve(input_array, window_size):
    ret = np.pad(input_array, (window_size - 1, 0), 'constant', constant_values=(0, 0))
    ret = np.convolve(ret, np.ones(window_size), 'valid')

    for i in range(1, window_size):
        ret[i - 1] = ret[i - 1] / i
    ret[window_size - 1:] = ret[window_size - 1:] / window_size

    return ret


def panPeakDetect(detection, fs):
    min_distance = int(0.25 * fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(len(detection)):

        if i > 0 and i < len(detection) - 1:
            if detection[i - 1] < detection[i] and detection[i + 1] < detection[i]:
                peak = i
                peaks.append(i)

                if detection[peak] > threshold_I1 and (peak - signal_peaks[-1]) > 0.3 * fs:

                    signal_peaks.append(peak)
                    indexes.append(index)
                    SPKI = 0.125 * detection[signal_peaks[-1]] + 0.875 * SPKI
                    if RR_missed != 0:
                        if signal_peaks[-1] - signal_peaks[-2] > RR_missed:
                            missed_section_peaks = peaks[indexes[-2] + 1:indexes[-1]]
                            missed_section_peaks2 = []
                            for missed_peak in missed_section_peaks:
                                if missed_peak - signal_peaks[-2] > min_distance and signal_peaks[
                                    -1] - missed_peak > min_distance and detection[missed_peak] > threshold_I2:
                                    missed_section_peaks2.append(missed_peak)

                            if len(missed_section_peaks2) > 0:
                                missed_peak = missed_section_peaks2[np.argmax(detection[missed_section_peaks2])]
                                missed_peaks.append(missed_peak)
                                signal_peaks.append(signal_peaks[-1])
                                signal_peaks[-2] = missed_peak

                else:
                    noise_peaks.append(peak)
                    NPKI = 0.125 * detection[noise_peaks[-1]] + 0.875 * NPKI

                threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                threshold_I2 = 0.5 * threshold_I1

                if len(signal_peaks) > 8:
                    RR = np.diff(signal_peaks[-9:])
                    RR_ave = int(np.mean(RR))
                    RR_missed = int(1.66 * RR_ave)

                index = index + 1

    signal_peaks.pop(0)

    return signal_peaks
