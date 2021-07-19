import sys
import os
from helper_code import *
import random


if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the data and model folders as arguments, e.g., python train_model.py data model.')

    data_directory = sys.argv[1]
    test_directory = sys.argv[2]
    
    
    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Extract classes from dataset.
    print('Extracting classes...')

    classes = set()
    class_to_file = {}
    for i, header_file in enumerate(header_files):
        header = load_header(header_file)
        single_class = get_labels(header)
        classes |= set(get_labels(header))
        
        for c in single_class:
            if c in class_to_file:
                class_to_file[c] += [[recording_files[i], header_file]]
            else:
                class_to_file[c] = [[recording_files[i], header_file]]
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)

    for key, value in class_to_file.items():
        c = int(len(value) * 0.1)
        test_data = random.sample(value, c)
        print(c)
       
        
        for mat, hea in test_data:
            if not (os.path.isfile(mat) and os.path.isfile(hea)):
                continue
                
            
            mat_new = mat.replace("training", "test")
            hea_new = hea.replace("training", "test")
            os.rename(mat, mat_new)
            os.rename(hea, hea_new)
        
        
    
    
    