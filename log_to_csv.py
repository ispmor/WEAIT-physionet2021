import argparse
import re

def get_numbers(line):
    return re.findall(r"[-+]?(?:\d*\.\d+|\d+)", line)

def build_csv(lines):
    all_time = []
    all_fmeasure = []
    all_acc = []
    all_challenge = []

    last_lead = 0
    dict_results = {}

    for idx, line in enumerate(lines):
        if "#" in line:
            fold_and_lead = get_numbers(line)
            avg_time = get_numbers(lines[idx+1])[0]
            f_measure = get_numbers(lines[idx+2])[0]
            acc = get_numbers(lines[idx+3])[0]
            challenge = get_numbers(lines[idx+4])[0]
            if fold_and_lead[1] in dict_results:
                dict_results[fold_and_lead[1]][fold_and_lead[0]] = [avg_time,
                                                                    f_measure,
                                                                    acc,
                                                                    challenge]
            else:
                dict_results[fold_and_lead[1]] = {fold_and_lead[0]:[avg_time,
                                                                    f_measure,
                                                                    acc,
                                                                    challenge]}

    print(dict_results)
    for fold in ['0','1','2','3','4']:
        fold_time = []
        fold_fmeasure = []
        fold_acc = []
        fold_challenge = []
        for lead in ['12','6','4','3','2']:
            fold_time.append(dict_results[lead][fold][0])
            fold_fmeasure.append(dict_results[lead][fold][1])
            fold_acc.append(dict_results[lead][fold][2])
            fold_challenge.append(dict_results[lead][fold][3])
        all_time.append(fold_time)
        all_fmeasure.append(fold_fmeasure)
        all_acc.append(fold_acc)
        all_challenge.append(fold_challenge)

    return {"time":all_time, "fmeasure":all_fmeasure, "acc":all_acc,
            "challenge":all_challenge}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some logs to csv.')
    parser.add_argument('path', type=str,
                        help='path to raw logs file')
    parser.add_argument('output', type=str,
                        help='output for preparsed logs file')

    args = parser.parse_args()

    final = []
    trening_final = []
    
    with open(args.path) as f:
        for l in f.readlines():
            if (
                    (len(l) < 70) and
                ("print" not in l) and (
                    ("LSTM " in l) or
                    ("GRU " in l) or
                    ("NBEATS " in l) or
                    ("#####   Fold=" in l) or
                    ("--- AVG peak classification time:" in l) or
                    ("--- F-measure:" in l) or
                    ("--- Accuracy: " in l) or
                    ("--- Challenge metric:" in l)
                )
            ):
                final.append(l)
                print(l)
            elif ( 
                    (len(l) < 70) and
                ("print" not in l) and (
                    ("LSTM " in l) or
                    ("GRU " in l) or
                    ("NBEATS " in l) or
                    ("##### TRENING  Fold=" in l) or
                    ("--- TRENING AVG peak classification time:" in l) or
                    ("--- TRENING F-measure:" in l) or
                    ("--- TRENING Accuracy: " in l) or
                    ("--- TRENING Challenge metric:" in l)
                )
            ):
                trening_final.append(l)
                print(l)

    official_csv_object = build_csv(final)
    training_csv_object = build_csv(trening_final)
    
    with open(args.output, 'w') as f:
        for key, value in official_csv_object.items():
             f.write(key)
             f.write("\n")
             for line in value:
                 print(line)
                 prep_val = ','.join(line)+"\n"
                 f.write(prep_val)
             f.write("\n")

        f.write("\n\n")
        f.write("TRENING\n")
        for key, value in official_csv_object.items():
             f.write(key)
             f.write("\n")
             for line in value:
                 f.write(','.join(line)+"\n")
             f.write("\n")

    
