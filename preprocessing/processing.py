# get waveform, label and record_id
import wfdb
import os
from tqdm import tqdm
import pandas as pd
import torch

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

for record in os.listdir(waveform_path):
    record_path = os.path.join(waveform_path, record)

    event_id_set = set()
    for event in os.listdir(record_path):
        event_id_set.add( os.path.splitext(event)[0])

    for event_id in event_id_set:
        event_path = os.path.join(record_path, event_id)
        sample_record = wfdb.rdrecord(event_path).p_signal  #Todo: (90000, 4), (90000, 5), (90000, 6) how many leads use or store
        sample_name = wfdb.rdrecord(event_path).record_name
        print(wfdb.rdrecord(event_path).sig_name)

        # print(sample_record.shape)

        # split train, val and test
        split_idx = split_events.index(sample_name)
        if split_splits[split_idx] == "train":
            train_samples.append(sample_record)
            train_names.append(sample_name)
        elif split_splits[split_idx] == "val":
            val_samples.append(sample_record)
            val_names.append(sample_name)
        else:
            test_samples.append(sample_record)
            test_names.append(sample_name)

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
    else:
        idx = test_names.index(event)
        test_ys[idx] = decision


# save pt
train_samples = torch.tensor(train_samples)
val_samples = torch.tensor(val_samples)
test_samples = torch.tensor(test_samples)

train_names = torch.tensor(train_names)
val_names = torch.tensor(val_names)
test_names = torch.tensor(test_names)

train_ys = torch.tensor(train_ys)
val_ys = torch.tensor(val_ys)
test_ys = torch.tensor(test_ys)

# Save the updated file with decisions
output_dir = r"D:\database\vtac\out\lead_selected"
for split in ["train", "val", "test"]:
    torch.save((train_samples, train_ys, train_names), f"{output_dir}/{split}.pt")

    print(f"Updated dataset saved at {output_dir}/{split}.pt")