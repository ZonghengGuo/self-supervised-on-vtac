# get waveform, label and record_id
import wfdb
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

dataset_path = r"D:\database\vtac"

samples = []
ys = []
names = []

# waveform and record_name
waveform_path = os.path.join(dataset_path, "waveforms")

for record in tqdm(os.listdir(waveform_path)):
    record_path = os.path.join(waveform_path, record)

    event_id_set = set()
    for event in os.listdir(record_path):
        event_id_set.add( os.path.splitext(event)[0])

    for event_id in event_id_set:
        event_path = os.path.join(record_path, event_id)
        sample_record = wfdb.rdrecord(event_path).p_signal
        sample_name = wfdb.rdrecord(event_path).record_name

        samples.append(sample_record)
        names.append(sample_name)

# get names and samples respectively
# get the label(order by )

# get label
event_labels_path = os.path.join(dataset_path, "event_labels.csv")
df = pd.read_csv(event_labels_path)

events = df["event"].astype(str).tolist()
decisions = df["decision"].astype(str).tolist()

for event, decision in zip(events, decisions):
    idx = names.index(event)  # Get index in names
    print(event, idx)
    ys[idx] = decision  # Assign corresponding decision













# # sample_record = wfdb.rdrecord(path)
# pprint(vars(sample_record))
# fs = sample_record.fs
# length = sample_record.sig_len
# record = sample_record.p_signal
#
# print(f"The record has {length//fs} minutes")
# print(record)

