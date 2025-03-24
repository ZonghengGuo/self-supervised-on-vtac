# get waveform, label and record_id
import wfdb
import os
from tqdm import tqdm
import pandas as pd
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
split_path = os.path.join(dataset_path, "benchmark_data_split")
split_pd = pd.read_csv(split_path)
split_events = split_pd["event"].astype(str).tolist()
split_splits = split_pd["split"].astype(str).tolist()

for record in tqdm(os.listdir(waveform_path)):
    record_path = os.path.join(waveform_path, record)

    event_id_set = set()
    for event in os.listdir(record_path):
        event_id_set.add( os.path.splitext(event)[0])

    for event_id in event_id_set:
        event_path = os.path.join(record_path, event_id)
        sample_record = wfdb.rdrecord(event_path).p_signal
        sample_name = wfdb.rdrecord(event_path).record_name

        #
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
ys = [0] * len(names)

event_labels_path = os.path.join(dataset_path, "event_labels.csv")
df = pd.read_csv(event_labels_path)

events = df["event"].astype(str).tolist()
decisions = df["decision"].astype(str).tolist()

for event, decision in zip(events, decisions):
    idx = names.index(event)  # Get index in names
    ys[idx] = decision  # Assign corresponding decision

# get split according to name





