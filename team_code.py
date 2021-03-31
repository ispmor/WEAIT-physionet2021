#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib



####
from torch.utils.tensorboard import SummaryWriter
from nbeats_pytorch.model import NBeatsNet
from torch.nn import functional as F
from torch import nn
import nbeats_additional_functions_2021 as naf
import os
import torch
from torch import optim
from config import exp_net_params as exp
from config import epoch_limit
from config import leads_dict 
import sys


twelve_lead_model_filename = '12_lead_model.th'
six_lead_model_filename = '6_lead_model.th'
three_lead_model_filename = '3_lead_model.th'
two_lead_model_filename = '2_lead_model.th'

forecast_length = exp["forecast_length"]
batch_size = exp["batch_size"]
backcast_length = exp["backcast_length"]
hidden = exp["hidden_layer_units"]
nb_blocks_per_stack = exp["nb_blocks_per_stack"]
thetas_dim = exp["thetas_dim"]


cuda0 = torch.cuda.set_device(0)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.pin_memory=False



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
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)
    print(classes)

    # Extract features and labels from dataset.
    print('Extracting features and labels...')

    data = np.zeros((num_recordings, 14), dtype=np.float32) # 14 features: one feature for each lead, one feature for age, and one feature for sex   
    
    
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool) # One-hot encoding of classes
    #neptune.init('puszkarb/physionet2021')
    
    name = twelve_lead_model_filename
    
    #experiment = neptune.create_experiment(name=name+ f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}')
    experiment = SummaryWriter()


    checkpoint_name = name + "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
    training_checkpoint = name + "_training"+ "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"
    
    #################
    # Creating a model - will have to somehowe automate it#
    #################
    
    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                forecast_length= forecast_length,
                thetas_dims=thetas_dim,
                nb_blocks_per_stack=nb_blocks_per_stack,
                backcast_length=backcast_length,
                hidden_layer_units=hidden,
                share_weights_in_stack=False,
                device=device,
                classes=classes)
    net.cuda()
    optimizer = optim.Adam(net.parameters())


#############################################3

    old_eval = 100
    
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        recording_full = get_leads_values(header, recording, twelve_leads)
        current_labels = get_labels(header)
        freq = get_frequency(header)
        if freq != float(500):
            recording_full = naf.equalize_signal_frequency(freq, recording_full)
            
            
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1

        new_eval = perform_training(net, optimizer, recording_full, forecast_length, backcast_length, batch_size, device, experiment, training_checkpoint, model_directory, labels[i], old_eval)
        if new_eval < old_eval:
            old_eval = new_eval
        
    

    # Train 12-lead ECG model.
    leads = twelve_leads
    filename = os.path.join(model_directory, twelve_lead_model_filename)

    feature_indices = [twelve_leads.index(lead) for lead in leads] #+ [12, 13]
    features = data[:, feature_indices]

    #imputer = SimpleImputer().fit(features)
    #features = imputer.transform(features)
    #classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    print('Savining 12-lead ECG model...')
    print(filename)
    save(filename, net, optimizer, classes, leads)
    
    

    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    leads = six_leads
    filename = os.path.join(model_directory, six_lead_model_filename)
    
    name = six_lead_model_filename
    


    checkpoint_name = name + "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
    training_checkpoint = name + "_training"+ "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"
    
    #################
    # Creating a model - will have to somehowe automate it#
    #################
    
    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                forecast_length= forecast_length,
                thetas_dims=thetas_dim,
                nb_blocks_per_stack=nb_blocks_per_stack,
                backcast_length=backcast_length,
                hidden_layer_units=hidden,
                share_weights_in_stack=False,
                device=device,
                classes=classes)
    net.cuda()
    optimizer = optim.Adam(net.parameters())


#############################################3

    old_eval = 100
    
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        recording_full = get_leads_values(header, recording, six_leads)
        freq = get_frequency(header)
        if freq != float(500):
            recording_full = naf.equalize_signal_frequency(freq, recording_full)
        current_labels = get_labels(header)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1

        print("Attempt to perform training")
        
        new_eval = perform_training(net, optimizer, recording_full, forecast_length, backcast_length, batch_size, device, experiment, training_checkpoint, model_directory, labels[i], old_eval)
        if new_eval < old_eval:
            old_eval = new_eval

    #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    #features = data[:, feature_indices]
    #imputer = SimpleImputer().fit(features)
    #features = imputer.transform(features)
    #classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    print('Savining 6-lead ECG model...')
    print(filename)
    save(filename, net, optimizer, classes, leads)
    
    
    
    
    
    
    
    

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    leads = three_leads
    filename = os.path.join(model_directory, three_lead_model_filename)
    name = three_lead_model_filename
    

    checkpoint_name = name + "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
    training_checkpoint = name + "_training"+ "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"
    
    #################
    # Creating a model - will have to somehowe automate it#
    #################
    
    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                forecast_length= forecast_length,
                thetas_dims=thetas_dim,
                nb_blocks_per_stack=nb_blocks_per_stack,
                backcast_length=backcast_length,
                hidden_layer_units=hidden,
                share_weights_in_stack=False,
                device=device,
                classes=classes)
    net.cuda()
    optimizer = optim.Adam(net.parameters())


#############################################3

    old_eval = 100
    
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        recording_full = get_leads_values(header, recording, three_leads)
        current_labels = get_labels(header)
        freq = get_frequency(header)
        if freq != float(500):
            recording_full = naf.equalize_signal_frequency(freq, recording_full)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1

        print("Attempt to perform training")
        
        new_eval = perform_training(net, optimizer, recording_full, forecast_length, backcast_length, batch_size, device, experiment, training_checkpoint, model_directory, labels[i], old_eval)
        if new_eval < old_eval:
            old_eval = new_eval

    #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    #features = data[:, feature_indices]
    #imputer = SimpleImputer().fit(features)
    #features = imputer.transform(features)
    #classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    print('Savining 3-lead ECG model...')
    print(filename)
    save(filename, net, optimizer, classes, leads)
    
    
    
    
 
    
    
    
   

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

    leads = two_leads
    filename = os.path.join(model_directory, two_lead_model_filename)
    name = two_lead_model_filename
    

    checkpoint_name = name + "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}'
    training_checkpoint = name + "_training"+ "_" + f'bl{nb_blocks_per_stack}-f{forecast_length}-b{backcast_length}-btch{batch_size}-h{hidden}' + ".th"
    
    #################
    # Creating a model - will have to somehowe automate it#
    #################
    
    net = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                forecast_length= forecast_length,
                thetas_dims=thetas_dim,
                nb_blocks_per_stack=nb_blocks_per_stack,
                backcast_length=backcast_length,
                hidden_layer_units=hidden,
                share_weights_in_stack=False,
                device=device,
                classes=classes)
    net.cuda()
    optimizer = optim.Adam(net.parameters())


#############################################3

    old_eval = 100
    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        recording_full = get_leads_values(header, recording, two_leads)
        current_labels = get_labels(header)
        freq = get_frequency(header)
        if freq != float(500):
            recording_full = naf.equalize_signal_frequency(freq, recording_full)
        for label in current_labels:
            if label in classes:
                j = classes.index(label)
                labels[i, j] = 1

        print("Attempt to perform training")
        
        new_eval = perform_training(net, optimizer, recording_full, forecast_length, backcast_length, batch_size, device, experiment, training_checkpoint, model_directory, labels[i], old_eval)
        if new_eval < old_eval:
            old_eval = new_eval

    #feature_indices = [twelve_leads.index(lead) for lead in leads] + [12, 13]
    #features = data[:, feature_indices]
    #imputer = SimpleImputer().fit(features)
    #features = imputer.transform(features)
    #classifier = RandomForestClassifier(n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)
    print('Savining 2-lead ECG model...')
    print(filename)
    save(filename, net, optimizer, classes, leads)
    

################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save(checkpoint_name, model, optimiser, classes, leads):
    torch.save({
        'classes': classes,
        'leads': leads,
        'model_state_dict': model.state_dict(),
        'optimiser_state_dict': optimiser.state_dict()
    }, checkpoint_name)


# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    filename = os.path.join(model_directory, twelve_lead_model_filename)
    return load_model(filename)

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    filename = os.path.join(model_directory, six_lead_model_filename)
    return load_model(filename)

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    filename = os.path.join(model_directory, three_lead_model_filename)
    return load_model(filename)

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    filename = os.path.join(model_directory, two_lead_model_filename)
    return load_model(filename)

# Generic function for loading a model.
def load_model(filename):
    checkpoint = torch.load(filename, map_location=torch.device('cuda:0'))

    model = NBeatsNet(stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK],
                forecast_length= forecast_length,
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

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording):
    classes = model.classes
    leads = model.leads

    features = get_leads_values(header, recording, leads)

    features = naf.one_file_training_data(features, forecast_length, backcast_length, device)
    # Predict labels and probabilities.
    _, probabilities = model(features.clone().detach())
 
    labels = np.asarray(probabilities.detach().cpu().numpy(), dtype=np.int)

    #probabilities = classifier.predict_proba(features)
    #probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

    return classes, labels, probabilities.detach().cpu().numpy()

################################################################################
#
# Other functions
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
    available_leads = get_leads(header)
    indices = list()
    for lead in leads:
        i = available_leads.index(lead)
        indices.append(i)
    recording = recording[indices, :]

    # Pre-process recordings.
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

    # Compute the root mean square of each ECG lead signal.
    rms = np.zeros(num_leads, dtype=np.float32)
    for i in range(num_leads):
        x = recording[i, :]
        rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

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
    adc_gains = get_adcgains(header, leads)
    baselines = get_baselines(header, leads)
    num_leads = len(leads)
    for i in range(num_leads):
        recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]
   
    return recording 


def train_full_grad_steps(data, device, net, optimizer, test_losses, training_checkpoint, size):
    global_step = naf.load(training_checkpoint, net, optimizer)
    local_step = 0
    each_epoch_plot = True

    for x_train_batch, y_train_batch in data:
        global_step += 1
        local_step += 1
        optimizer.zero_grad()
        net.train()
        _, forecast = net(x_train_batch.clone().detach())#.to(device)) #Dodaje od 
        m = nn.BCEWithLogitsLoss()

        loss = m(forecast, y_train_batch[0])#torch.zeros(size=(16,)))
        #loss = F.mse_loss(forecast, y_train_batch.clone().detach())#.to(device))
        loss.backward()
        optimizer.step()
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                print("Training batches passed: %d" % (local_step))
                naf.save(training_checkpoint, net, optimizer, global_step)
        if local_step > 0 and local_step % size == 0:
            return global_step



def perform_training(net, optimizer, recordings, forecast_length, backcast_length, batch_size, device, experiment, training_checkpoint, model_directory, labels, old_eval):
    test_losses = []
    the_lowest_error = [100]

    
    data, x_train, y_train, x_test, y_test = naf.get_data_with_labels(recordings,forecast_length, backcast_length, batch_size, device, labels)
    

    global_step = train_full_grad_steps(data, device, net, optimizer, test_losses, model_directory + training_checkpoint, x_train.shape[0])

    train_eval = naf.evaluate_training(backcast_length,
                                       forecast_length,
                                       net,
                                       test_losses,
                                       x_train,
                                       y_train,
                                       the_lowest_error,
                                       device,
                                       experiment=experiment)
                                                                                                  
    experiment.add_scalar(f'train_loss_{training_checkpoint}', train_eval)


    new_eval = naf.evaluate_training(backcast_length,
                                     forecast_length,
                                     net,
                                     test_losses,
                                     x_test,
                                     y_test, 
                                     the_lowest_error,
                                     device,
                                     experiment=experiment)
    experiment.add_scalar(f'eval_loss_{training_checkpoint}', new_eval)
    
    print("\n New evaluation sccore: %f, ---->>>> old score: %f" % (new_eval, old_eval))
    #if new_eval < old_eval:
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
