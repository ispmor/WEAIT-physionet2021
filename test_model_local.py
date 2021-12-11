#!/usr/bin/env python

# Do *not* edit this script.

import numpy as np, os, sys
#from team_code import load_model, run_model
from helper_code import *

classes_glob = []

import os, os.path, sys, numpy as np
from helper_code import get_labels, is_finite_number, load_header, load_outputs
#import seaborn as sn
#import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve
# Test model.
def test_model(model_directory, data_directory, output_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)

    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

        # Load the scored classes and the weights for the Challenge metric.
    print('Loading weights...')
    weights_file = '/home/puszkar/evaluation-2021/weights.csv'
    sinus_rhythm = set(['426783006'])
    classes, weights = load_weights(weights_file)
    global classes_glob
    classes_glob = classes

    labels = load_labels(header_files, classes)

    binary_outputs = []
    scalar_outputs = np.ndarray((num_recordings, 26))
    c = np.ndarray((num_recordings, 26))

    # Create a folder for the outputs if it does not already exist.
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Identify the required lead sets.
    required_lead_sets = set()
    for i in range(num_recordings):
        header = load_header(header_files[i])
        leads = get_leads(header)
        required_lead_sets.add(sort_leads(leads))

    # Load models.
    leads_to_model = dict()
    print('Loading models...')
    for leads in required_lead_sets:
        model = load_model(model_directory, leads)  ### Implement this function!
        leads_to_model[leads] = model

    std_array = []
    # Run model for each recording.
    print('Running model...')
    returned_classes = None


    for i in range(num_recordings):
        print('    {}/{}...'.format(i + 1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        leads = get_leads(header)

        # Apply model to recording.
        model = leads_to_model[leads]
        c[i], binary_outputs, scalar_outputs[i] = run_model(model, header, recording)  ### Implement this function!
        # Save model outputs.

    thresholds_per_class = [0.94649589, 0.75176591, 0.92429638, 0.97407532, 0.92660624, 0.70162058, 0.79427606,
                            0.99882579, 0.81119329, 0.94413322, 0.99160779, 0.98631829,
                            0.81352568, 0.81511146, 0.68745172, 0.84047282, 1., 0.89702266, 0.63237596, 0.93686378,
                            0.9980495, 0.91167629, 0.71462607, 0.6771611, 0.94447708, 0.94156897]
    print("------------------------ optim?")
    binary_outputs = [[scalar_outputs[i][j] > t for j, t in enumerate(thresholds_per_class)] for i in
                      range(len(scalar_outputs))]
    binary_outputs_local, scalar_outputs_local = load_classifier_outputs(binary_outputs, scalar_outputs, c, classes_glob)
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs_local)
    accuracy = compute_accuracy(labels, binary_outputs_local)
    print('--- Accuracy: ', accuracy)

    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs_local)
    print('--- F-measure: ', f_measure)

    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs_local, classes, sinus_rhythm)
    print('--- Challenge metric: ', challenge_metric)

    thresholds = np.linspace(0.6, 0.85, 25)
    for thr in thresholds:
        print("\n\nTHR: ", thr)
        binary_outputs = [scalar_outputs[i] > thr for i in range(len(scalar_outputs))]
        binary_outputs_local, scalar_outputs_local = load_classifier_outputs(binary_outputs, scalar_outputs, c, classes_glob)
        auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs_local)
        accuracy = compute_accuracy(labels, binary_outputs_local)
        print('--- Accuracy: ', accuracy)

        f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs_local)
        print('--- F-measure: ', f_measure)

        challenge_metric = compute_challenge_metric(weights, labels, binary_outputs_local, classes, sinus_rhythm)
        print('--- Challenge metric: ', challenge_metric)

        #recording_id = get_recording_id(header)
        #head, tail = os.path.split(header_files[i])
        #root, extension = os.path.splitext(tail)
        #output_file = os.path.join(output_directory, root + '.csv')
        #save_outputs(output_file, recording_id, classes, labels, probabilities)




    print('Done.')


def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table) - 1
    if num_rows < 1:
        raise Exception('The table {} is empty.'.format(table_file))
    row_lengths = set(len(table[i]) - 1 for i in range(num_rows))
    if len(row_lengths) != 1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(row_lengths)
    if num_cols < 1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j + 1] for j in range(num_rows)]
    cols = [table[i + 1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols), dtype=np.float64)
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i + 1][j + 1]
            if is_finite_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values


# Load weights.
def load_weights(weight_file):
    # Load the table with the weight matrix.
    rows, cols, values = load_table(weight_file)

    # Split the equivalent classes.
    rows = [set(row.split('|')) for row in rows]
    cols = [set(col.split('|')) for col in cols]
    assert (rows == cols)

    # Identify the classes and the weight matrix.
    classes = rows
    weights = values

    return classes, weights


# Load labels from header/label files.
def load_labels(label_files, classes):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)
    num_classes = len(classes)

    # Use one-hot encoding for the labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)

    # Iterate over the recordings.
    for i in range(num_recordings):
        header = load_header(label_files[i])
        y = set(get_labels(header))
        for j, x in enumerate(classes):
            if x & y:
                labels[i, j] = 1

    return labels


# Load outputs from output files.
def load_classifier_outputs(recording_binary, recording_scalars, recording_c, classes_glob):
    # The outputs should have the following form:
    #
    # #Record ID
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           1
    #        0.12,        0.34,        0.56
    #

    num_recordings = len(recording_scalars)
    num_classes = len(classes_glob)


    # Use one-hot encoding for the outputs.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)

    # Iterate over the recordings.
    for i in range(num_recordings):
        recording_id, recording_classes, recording_binary_outputs, recording_scalar_outputs = i, recording_c[i], recording_binary[i], recording_scalars[i]

        # Allow for equivalent classes and sanitize classifier outputs.
        recording_binary_outputs = [1 if ((is_finite_number(entry) and float(entry)==1) or (entry in ('True', 'true', 'T', 't'))) else 0 for entry in recording_binary_outputs]
        recording_scalar_outputs = [float(entry) if is_finite_number(entry) else 0 for entry in recording_scalar_outputs]
        recording_classes = [{str(int(r))} for r in recording_classes]
        # Allow for unordered/reordered and equivalent classes.
        for j, x in enumerate(classes_glob): #<- global classes
            binary_values = list()
            scalar_values = list()
            for k, y in enumerate(recording_classes):# <- recording classes
                if x & y:
                    binary_values.append(recording_binary_outputs[k])
                    scalar_values.append(recording_scalar_outputs[k])
            if binary_values:
                binary_outputs[i, j] = any(binary_values) # Define a class as positive if any of the equivalent classes is positive.
            if scalar_values:
                scalar_outputs[i, j] = np.mean(scalar_values) # Define the scalar value of a class as the mean value of the scalar values across equivalent classes.

    return binary_outputs, scalar_outputs



# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :] == outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)


# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1
                else:  # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(max(np.sum(labels[i, :]), 1))
            for j in range(num_classes):
                if labels[i, j] == 1 and outputs[i, j] == 1:  # TP
                    A[j, 1, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 1:  # FP
                    A[j, 1, 0] += 1.0 / normalization
                elif labels[i, j] == 1 and outputs[i, j] == 0:  # FN
                    A[j, 0, 1] += 1.0 / normalization
                elif labels[i, j] == 0 and outputs[i, j] == 0:  # TN
                    A[j, 0, 0] += 1.0 / normalization
                else:  # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A


# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)
    #global classes_glob
    #for i, matrice in enumerate(A):
        #df_cm = pd.DataFrame(matrice, index=['-', '+'], columns=['-', '+'])
        #plt.figure(figsize=(10, 7))
        #sn.heatmap(df_cm, annot=True)

        #name = "./plots/" + ''.join(classes_glob[i]) + "confusion_matrix_lstm-20-08.png"
        #plt.savefig(name)
        #plt.close()

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    if np.any(np.isfinite(f_measure)):
        macro_f_measure = np.nanmean(f_measure)
    else:
        macro_f_measure = float('nan')

    return macro_f_measure, f_measure


# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)
    micro = np.zeros(num_classes)
    macro = np.zeros(num_classes)
    optim = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1] + 1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k] == 1)
        tn[0] = np.sum(labels[:, k] == 0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j - 1]
            fp[j] = fp[j - 1]
            fn[j] = fn[j - 1]
            tn[j] = tn[j - 1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        fpr = np.zeros(num_thresholds)

        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
                fpr[j] = float(fp[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
                fpr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds - 1):
            auroc[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]

    fpr2 = dict()
    tpr2 = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr2[i], tpr2[i], thresholds = roc_curve(labels[:, i], outputs[:, i])
        roc_auc[i] = auc(fpr2[i], tpr2[i])

    fpr2["micro"], tpr2["micro"], thresholds = roc_curve(labels.ravel(), outputs.ravel())
    roc_auc["micro"] = auc(fpr2["micro"], tpr2["micro"])

    #for j in range(len(thresholds)):
        #if 0.795 < tpr2["micro"][j] < 0.805:
            #print("Threshold: ", thresholds[j])

    all_fpr = np.unique(np.concatenate([fpr2[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr2[i], tpr2[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    fpr2["macro"] = all_fpr
    tpr2["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr2["macro"], tpr2["macro"])

    # Compute macro AUROC and macro AUPRC across classes.
    if np.any(np.isfinite(auroc)):
        macro_auroc = np.nanmean(auroc)
    else:
        macro_auroc = float('nan')
    if np.any(np.isfinite(auprc)):
        macro_auprc = np.nanmean(auprc)
    else:
        macro_auprc = float('nan')

    # title = 'micro-macro.png'
    # plt.figure()
    # plt.title(title)
    # plt.plot(fpr2["micro"], tpr2["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr2["macro"], tpr2["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.savefig(title)
    # plt.close()

    # global classes_glob
    # for k, c in enumerate(classes_glob):
    #     precision, recall, thresholds = precision_recall_curve(labels[:, k], outputs[:, k])
    #     fscore = (2 * precision * recall) / (precision + recall)
    #     # locate the index of the largest f score
    #     ix = np.argmax(fscore)
    #     optim[k] = thresholds[ix]
    #     #print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    #
    # print("THRESHOLD PER CLASS")
    # print(optim)
    return macro_auroc, macro_auprc, auroc, auprc


# Compute a modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        # Calculate the number of positive labels and/or outputs.
        normalization = float(max(np.sum(np.any((labels[i, :], outputs[i, :]), axis=0)), 1))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0 / normalization

    # global classes_glob
    # df_cm = pd.DataFrame(A, index=classes_glob, columns=classes_glob)
    # plt.figure(figsize=(12, 10))
    # sn.heatmap(df_cm, annot=True)
    # plt.savefig("confusion_matrix.png")
    # plt.close()
    return A


# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, sinus_rhythm):
    num_recordings, num_classes = np.shape(labels)
    if sinus_rhythm in classes:
        sinus_rhythm_index = classes.index(sinus_rhythm)
    else:
        raise ValueError('The sinus rhythm class is not available.')

    # Compute the observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute the score for the model that always chooses the sinus rhythm class.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, sinus_rhythm_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = 0.0

    return normalized_score


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception(
            'Include the model, data, and output folders as arguments, e.g., python test_model.py model data outputs.')

    model_directory = sys.argv[1]
    data_directory = sys.argv[2]
    output_directory = sys.argv[3]

    test_model(model_directory, data_directory, output_directory)
