#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np

####
from torch.utils.tensorboard import SummaryWriter
from nbeats_pytorch.model import NBeatsNet
from torch import nn
from torch.utils import data as torch_data
import nbeats_additional_functions_2021 as naf
import os
import torch
from torch import optim
from config import exp_net_params as exp
import h5py
from h5class import HDF5Dataset

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
leads_set = (twelve_leads, six_leads, four_leads, three_leads, two_leads)

single_peak_length = exp["single_peak_length"]
forecast_length = exp["forecast_length"]
batch_size = exp["batch_size"]
backcast_length = exp["backcast_length"]
hidden = exp["hidden_layer_units"]
nb_blocks_per_stack = exp["nb_blocks_per_stack"]
thetas_dim = exp["thetas_dim"]

cuda0 = torch.cuda.set_device(0)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.pin_memory = False


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
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x))  # Sort classes numerically if numbers.
    else:
        classes = sorted(classes)  # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    print(classes)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data = np.zeros((num_recordings, 14),
                    dtype=np.float32)  # 14 features: one feature for each lead, one feature for age, and one feature for sex

    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)  # One-hot encoding of classes

    for leads in leads_set:

        name = get_model_filename(leads)

        experiment = SummaryWriter()

        checkpoint_name = name[
                          :-3] + "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
        training_checkpoint = name[
                              :-3] + "_training" + "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"

        net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                        forecast_length=forecast_length,
                        thetas_dims=thetas_dim,
                        nb_blocks_per_stack=nb_blocks_per_stack,
                        backcast_length=backcast_length,
                        hidden_layer_units=hidden,
                        share_weights_in_stack=False,
                        device=device,
                        classes=classes)
        net.cuda()
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        init_dataset = list(range(num_recordings))
        lengths = [int(len(init_dataset) * 0.8), len(init_dataset) - int(len(init_dataset) * 0.8)]
        data_training, data_validation = torch_data.random_split(init_dataset, lengths)

        ################ CREATE HDF5 DATABASE #############################3
        if not os.path.isfile(f'cinc_database_{len(leads)}_training.h5'):
            create_hdf5_db(data_training, num_classes, header_files, recording_files, classes, leads, isTraining=True)
        if not os.path.isfile(f'cinc_database_{len(leads)}_validation.h5'):
            create_hdf5_db(data_training, num_classes, header_files, recording_files, classes, leads, isTraining=False)

        training_dataset = HDF5Dataset('./' + f'cinc_database_{len(leads)}_training.h5', recursive=False,
                                       load_data=False,
                                       data_cache_size=4, transform=None)
        validation_dataset = HDF5Dataset('./' + f'cinc_database_{len(leads)}_validation.h5', recursive=False,
                                         load_data=False,
                                         data_cache_size=4, transform=None)

        training_data_loader = torch_data.DataLoader(training_dataset, batch_size=20000, shuffle=True, num_workers=6)
        validation_data_loader = torch_data.DataLoader(validation_dataset, batch_size=20000, shuffle=True, num_workers=6)

        print("data_loader", training_data_loader)
        num_epochs = 1000
        m = nn.BCEWithLogitsLoss()
        for epoch in range(num_epochs):
            local_step = 0
            epoch_loss = []
            for x, y in training_data_loader:
                local_step += 1
                optimizer.zero_grad()
                net.train()

                _, forecast = net(x.to(device))  # .to(device)) #Dodaje od
                loss = m(forecast, y.to(device))  # torch.zeros(size=(16,)))

                epoch_loss.append(loss)
                loss.backward()
                optimizer.step()

            mean = torch.mean(torch.stack(epoch_loss))
            print("Epoch: %d Training loss: %f" % (epoch, mean))
            experiment.add_scalar(f'train_loss', mean, epoch)

            with torch.no_grad():
                epoch_loss = []
                if epoch != 0 and epoch % 100 == 0:
                    for x, y in validation_data_loader:
                        net.eval()
                        _, forecast = net(x.to(device))  # .to(device)) #Dodaje od
                        m = nn.BCEWithLogitsLoss()
                        loss = m(forecast, y.to(device))  # torch.zeros(size=(16,)))
                        epoch_loss.append(loss)

                mean = torch.mean(torch.stack(epoch_loss))
                experiment.add_scalar(f'validation_loss', mean, epoch)
                print("Epoch: %d Validation loss: %f" % (epoch, mean))

                naf.save(training_checkpoint, net, optimizer, epoch)

            filename = os.path.join(model_directory, name)
            print(f'Savining {len(leads)}-lead ECG model, epoch: {epoch}...')
            save(filename, net, optimizer, classes, leads)


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

    with h5py.File(f'cinc_database_{len(leads)}_{group}.h5', 'w') as h5file:

        grp = h5file.create_group(group)

        dset = grp.create_dataset("data", (1, len(leads), single_peak_length),
                                  maxshape=(None, len(leads), single_peak_length), dtype='f',
                                  chunks=(1, len(leads), single_peak_length))
        lset = grp.create_dataset("label", (1, num_classes), maxshape=(None, num_classes), dtype='f',
                                  chunks=(1, num_classes))
        counter = 0
        for i in num_recordings:
            print('    {}/{}...'.format(counter + 1, len(num_recordings)))
            counter += 1
            # Load header and recording.
            header = load_header(header_files[i])
            recording = np.array(load_recording(recording_files[i]), dtype=np.float32)

            recording_full = get_leads_values(header, recording, leads)
            current_labels = get_labels(header)
            freq = get_frequency(header)
            if freq != float(500):
                recording_full = naf.equalize_signal_frequency(freq, recording_full)

            recording_full = naf.one_file_training_data(recording_full, single_peak_length, device)
            local_label = np.zeros((num_classes,), dtype=np.bool)
            for label in current_labels:
                if label in classes:
                    j = classes.index(label)
                    local_label[j]

            new_windows = recording_full.shape[0]
            dset.resize(dset.shape[0] + new_windows, axis=0)
            dset[-new_windows:] = recording_full

            label_pack = [local_label for i in range(recording_full.shape[0])]
            lset.resize(lset.shape[0] + new_windows, axis=0)
            lset[-new_windows:] = label_pack

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

    model = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                      forecast_length=forecast_length,
                      thetas_dims=thetas_dim,
                      nb_blocks_per_stack=nb_blocks_per_stack,
                      backcast_length=backcast_length,
                      hidden_layer_units=hidden,
                      share_weights_in_stack=False,
                      device=device,
                      classes=checkpoint['classes'])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.leads = checkpoint['leads']
    model.cuda()
    print(f'Restored checkpoint from {filename}.')
    return model


# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model.classes
    leads = model.leads

    features = get_leads_values(header, recording, leads)

    features = naf.one_file_training_data(features, forecast_length, backcast_length, device)
    # Predict labels and probabilities.
    _, probabilities = model(features.clone().detach())

    labels = np.asarray(probabilities.detach().cpu().numpy(), dtype=np.int)

    # probabilities = classifier.predict_proba(features)
    # probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities.detach().cpu().numpy()


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


def train_full_grad_steps(data, net, optimizer, training_checkpoint, size, global_step):
    global_step_checkpoint = naf.load(training_checkpoint, net, optimizer)
    print(f"Global step loaded from the checkpoint: {global_step_checkpoint}")
    local_step = 0
    each_epoch_plot = True

    for x_train_batch, y_train_batch in data:

        local_step += 1
        optimizer.zero_grad()
        net.train()
        _, forecast = net(x_train_batch.clone().detach())  # .to(device)) #Dodaje od
        m = nn.BCEWithLogitsLoss()

        loss = m(forecast, y_train_batch[0])  # torch.zeros(size=(16,)))
        # loss = F.mse_loss(forecast, y_train_batch.clone().detach())#.to(device))
        loss.backward()
        optimizer.step()

        if local_step > 0 and local_step % size == 0:
            break

    with torch.no_grad():
        print("Training batches passed: %d" % (local_step))
        print(f"Global step saved: {global_step}")

        naf.save(training_checkpoint, net, optimizer, global_step)

    if local_step > 0 and local_step % size == 0:
        return global_step


def perform_training(net, optimizer, recordings, forecast_length, backcast_length, batch_size, device, experiment,
                     training_checkpoint, model_directory, labels, old_eval, i):
    test_losses = []
    the_lowest_error = [100]

    data, x_train, y_train, x_test, y_test = naf.get_data_with_labels(recordings, forecast_length, backcast_length,
                                                                      batch_size, device, labels)

    global_step = train_full_grad_steps(data, device, net, optimizer, test_losses,
                                        model_directory + training_checkpoint, x_train.shape[0], i)

    train_eval = naf.evaluate_training(backcast_length,
                                       forecast_length,
                                       net,
                                       test_losses,
                                       x_train,
                                       y_train,
                                       the_lowest_error,
                                       device,
                                       experiment=experiment)

    experiment.add_scalar(f'train_loss_{training_checkpoint}', train_eval, i)

    new_eval = naf.evaluate_training(backcast_length,
                                     forecast_length,
                                     net,
                                     test_losses,
                                     x_test,
                                     y_test,
                                     the_lowest_error,
                                     device,
                                     experiment=experiment)
    experiment.add_scalar(f'eval_loss_{training_checkpoint}', new_eval, i)

    print("\n New evaluation sccore: %f, ---->>>> old score: %f" % (new_eval, old_eval))
    # if new_eval < old_eval:
    #    print("in if")
    #    difference = old_eval - new_eval
    #    old_eval = new_eval
    #    with torch.no_grad():
    #        if training_checkpoint:
    #            print("there is a checkpoint!" + training_checkpoint)
    #            os.remove(training_checkpoint)
    #        print(model_directory + training_checkpoint)
    #        naf.save(model_directory + training_checkpoint, net, optimizer, global_step)
    return new_eval
