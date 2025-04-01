# read records but exclude without "PLETH", "ABP", get the record_id_lists

import wfdb
import os
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np

dataset_path = "../../database/vtac"
save_path = dataset_path + "/out/raw"

# import matplotlib.pyplot as plt
#
# def plot_sample_record(sample_record, index_ppg, index_abp):
#     # Select the signals
#     signal_0 = sample_record[0:15000, 0]
#     signal_1 = sample_record[0:15000, 1]
#     pleth = sample_record[0:15000, index_ppg]
#     abp = sample_record[0:15000, index_abp]
#
#     # Plot the signals
#     plt.figure(figsize=(12, 8))
#
#     plt.subplot(4, 1, 1)
#     plt.plot(signal_0, label='Signal 0')
#     plt.title('Signal 0')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.legend()
#
#     plt.subplot(4, 1, 2)
#     plt.plot(signal_1, label='Signal 1')
#     plt.title('Signal 1')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.legend()
#
#     plt.subplot(4, 1, 3)
#     plt.plot(pleth, label='PLETH')
#     plt.title('PLETH')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.legend()
#
#     plt.subplot(4, 1, 4)
#     plt.plot(abp, label='ABP')
#     plt.title('ABP')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.show()

# get waveform and label
waveform_path = os.path.join(dataset_path, "waveforms")
csv_path = os.path.join(dataset_path, "event_labels.csv")
event_label_df = pd.read_csv(csv_path)

for record in tqdm(os.listdir(waveform_path)):
    record_path = os.path.join(waveform_path, record)

    event_id_set = set()
    for event in os.listdir(record_path):
        event_id_set.add(os.path.splitext(event)[0])

    for event_id in event_id_set:
        required_samples = []

        event_path = os.path.join(record_path, event_id)
        record = wfdb.rdrecord(event_path)

        sample_record = record.p_signal
        sample_name = record.record_name
        sample_length = record.sig_len
        sig_names = record.sig_name

        if "ABP" not in sig_names:
            continue

        if "PLETH" not in sig_names:
            continue

        if len(sig_names) < 4:
            continue

        # Set missing values to 0
        sample_record = np.nan_to_num(sample_record, nan=0.0)

        # Select the required channels
        index_ppg = sig_names.index("PLETH")
        index_abp = sig_names.index("ABP")

        required_samples.append(sample_record[:, 0])
        required_samples.append(sample_record[:, 1])
        required_samples.append(sample_record[:, index_ppg])
        required_samples.append(sample_record[:, index_abp])

        # plot_sample_record(sample_record, index_ppg, index_abp)

        required_samples = np.array(required_samples)

        # get label
        decision_value = event_label_df.loc[event_label_df['event'] == sample_name, 'decision'].values[0]

        if decision_value == False:
            decision_value = 0
        elif decision_value == True:
            decision_value = 1

        # Save sample_record and decision_value as .npy files
        np.save(f"{save_path}/{sample_name}_record.npy", required_samples)
        np.save(f"{save_path}/{sample_name}_label.npy", decision_value)