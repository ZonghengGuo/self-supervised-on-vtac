# get waveform, label and record_id
import wfdb
import os
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np

dataset_path = r"D:\database\vtac"

train_samples = []
val_samples = []
test_samples = []

train_names = []
val_names = []
test_names = []


# get waveform and record_name
waveform_path = os.path.join(dataset_path, "waveforms")

# find which split to store(train, val, test)
split_path = os.path.join(dataset_path, "benchmark_data_split.csv")
split_pd = pd.read_csv(split_path)
split_events = split_pd["event"].astype(str).tolist()
split_splits = split_pd["split"].astype(str).tolist()

for record in tqdm(os.listdir(waveform_path)):
    record_path = os.path.join(waveform_path, record)

    event_id_set = set()
    for event in os.listdir(record_path):
        event_id_set.add(os.path.splitext(event)[0])

    for event_id in event_id_set:
        required_samples = []

        event_path = os.path.join(record_path, event_id)
        sample_record = wfdb.rdrecord(event_path).p_signal  #Todo: (90000, 4), (90000, 5), (90000, 6) how many leads use or store
        sample_name = wfdb.rdrecord(event_path).record_name

        sig_names = wfdb.rdrecord(event_path).sig_name

        if "ABP" not in sig_names or "PLETH" not in sig_names or "II" not in sig_names or "V" not in sig_names:
            continue

        if wfdb.rdrecord(event_path).sig_len != 90000:
            print("The length not match")
            continue

        index_2 = sig_names.index("II")
        index_5 = sig_names.index("V")
        index_ppg = sig_names.index("PLETH")
        index_abp = sig_names.index("ABP")

        required_samples.append(sample_record[:, index_2])
        required_samples.append(sample_record[:, index_5])
        required_samples.append(sample_record[:, index_ppg])
        required_samples.append(sample_record[:, index_abp])

        # split train, val and test
        split_idx = split_events.index(sample_name)
        if split_splits[split_idx] == "train":
            train_samples.append(required_samples)
            train_names.append(sample_name)
        elif split_splits[split_idx] == "val":
            val_samples.append(required_samples)
            val_names.append(sample_name)
        else:
            test_samples.append(required_samples)
            test_names.append(sample_name)

print(f"the length of training is {len(train_samples)}, val is {len(val_samples)} and test is {len(test_samples)}")

# get label
train_ys = [0] * len(train_samples)
val_ys = [0] * len(val_samples)
test_ys = [0] * len(test_samples)

event_labels_path = os.path.join(dataset_path, "event_labels.csv")
df = pd.read_csv(event_labels_path)

events = df["event"].astype(str).tolist()
decisions = df["decision"].astype(str).tolist()

for event, decision in zip(events, decisions):
    if event in train_names:
        idx = train_names.index(event)
        train_ys[idx] = decision
    elif event in val_names:
        idx = val_names.index(event)
        val_ys[idx] = decision
    elif event in test_names:
        idx = test_names.index(event)
        test_ys[idx] = decision


# save pt
train_samples = torch.tensor(np.array(train_samples))
val_samples = torch.tensor(np.array(val_samples))
test_samples = torch.tensor(np.array(test_samples))

train_ys = torch.tensor([1 if x == 'True' else 0 for x in train_ys], dtype=torch.int)
val_ys = torch.tensor([1 if x == 'True' else 0 for x in val_ys], dtype=torch.int)
test_ys = torch.tensor([1 if x == 'True' else 0 for x in test_ys], dtype=torch.int)

# train_names = torch.tensor(np.array(train_names))
# val_names = torch.tensor(np.array(val_names))
# test_names = torch.tensor(np.array(test_names))

# Save the updated file with decisions
output_dir = r"D:\database\vtac\out\lead_selected"

torch.save((train_samples, train_ys, train_names), f"{output_dir}/train.pt")
print(f"Finish training dataset saved!!!")

torch.save((val_samples, val_ys, val_names), f"{output_dir}/val.pt")
print(f"Finish validating dataset saved!!!")

torch.save((test_samples, test_ys, test_names), f"{output_dir}/test.pt")
print(f"Finish testing dataset saved!!!")

