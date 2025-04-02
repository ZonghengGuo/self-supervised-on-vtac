# read records but exclude without "PLETH", "ABP", get the record_id_lists
import numpy as np
import wfdb
import os
from tqdm import tqdm
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


SAMPLING_FREQ = 250
POWERLINE_FREQ = 60

def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def notch_filter(freq, Q, fs):
    b, a = iirnotch(freq, Q, fs)
    return b, a


def filter_ecg_channel(data):
    b, a = butter_highpass(1.0, SAMPLING_FREQ)
    b2, a2 = butter_lowpass(30.0, SAMPLING_FREQ)
    tempfilt = filtfilt(b, a, data)
    tempfilt = filtfilt(b2, a2, tempfilt)
    b_notch, a_notch = notch_filter(POWERLINE_FREQ, 30, SAMPLING_FREQ)
    tempfilt = filtfilt(b_notch, a_notch, tempfilt)
    return tempfilt


def filter_ppg_channel(data):
    b_notch, a_notch = notch_filter(POWERLINE_FREQ, 30, SAMPLING_FREQ)
    tempfilt = filtfilt(b_notch, a_notch, data)
    N_bp, Wn_bp = butter(1, [0.5, 5], btype="band", analog=False, fs=SAMPLING_FREQ)
    tempfilt = filtfilt(N_bp, Wn_bp, tempfilt)
    return tempfilt


def filter_abp_channel(data):
    b_notch, a_notch = notch_filter(POWERLINE_FREQ, 30, SAMPLING_FREQ)
    tempfilt = filtfilt(b_notch, a_notch, data)
    b2, a2 = butter_lowpass(16.0, SAMPLING_FREQ)
    tempfilt = filtfilt(b2, a2, tempfilt)
    return tempfilt


def min_max_norm(data, feature_range=(0, 1)):
    min_val = np.min(data)
    max_val = np.max(data)

    if max_val == min_val:  # Avoid division by zero
        return np.zeros_like(data) if feature_range[0] == 0 else np.full_like(data, feature_range[0])

    scale = feature_range[1] - feature_range[0]
    return feature_range[0] + (data - min_val) * scale / (max_val - min_val)


dataset_path = "data"
save_path = dataset_path + "/out/raw"

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

        # At least needs 4 channels
        if len(sig_names) < 4:
            continue

        # Impute
        sample_record = np.nan_to_num(sample_record, nan=0.0)

        required_samples = []

        candidates = ["ABP", "PLETH", "II", "V", "aVR", "III", "I", "V2", "MCL", "aVF", "aVL"]
        test = []
        for candi_sig_name in candidates:
            if len(required_samples) < 4:
                if candi_sig_name in sig_names:
                    index_candi = sig_names.index(candi_sig_name)
                    single_lead = sample_record[:, index_candi]
                    if candi_sig_name == "ABP":
                        single_lead = filter_abp_channel(single_lead)
                    elif candi_sig_name == "PLETH":
                        single_lead = filter_ppg_channel(single_lead)
                    else:
                        single_lead = filter_ecg_channel(single_lead)
                    single_lead = min_max_norm(single_lead)
                    required_samples.append(single_lead)
            else:
                break

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