import wfdb
import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg


def peak_detection(sig, fs):
    r = ecg.ecg(signal=sig, sampling_rate=fs, show=False)
    return r['rpeaks']

# Min-max normalization
def min_max_normalize(signal, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    signal = np.asarray(signal)

    min_signal = np.min(signal)
    max_signal = np.max(signal)

    if max_signal == min_signal:
        return np.zeros_like(signal) if min_val == 0 else np.full_like(signal, min_val)

    normalized_signal = (signal - min_signal) / (max_signal - min_signal)
    normalized_signal = normalized_signal * (max_val - min_val) + min_val

    return normalized_signal

def set_nan_to_zero(sig):
    zero_segment = np.nan_to_num(sig, nan=0.0)
    return zero_segment

if __name__ == '__main__':
    database_name = 'mimic3wdb'
    rel_segment_name = '3000801_0003'
    rel_segment_dir = 'mimic3wdb/30/3000801'
    segment_data = wfdb.rdrecord(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    ecg_signal = segment_data.p_signal[:, 0][0:3750]
    sampling_rate = segment_data.fs

    print(np.where(np.isnan(ecg_signal)))

    # set nan as zero then normalize
    ecg_signal = set_nan_to_zero(ecg_signal)
    ecg_signal = min_max_normalize(ecg_signal)

    print(np.where(np.isnan(ecg_signal)))

    r_peaks = peak_detection(ecg_signal, sampling_rate)

    print(f"detect {len(r_peaks)} R peaks")
    print(f"R peaks index: {r_peaks}")

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(bottom=0.2)

    time = np.arange(len(ecg_signal))/sampling_rate
    line, = ax.plot(time, ecg_signal, label='ECG Signal')

    peaks_scatter = ax.scatter(r_peaks/sampling_rate, ecg_signal[r_peaks], color='red', s=50, label='R Peaks')

    ax.set_title('ECG Signal with Detected R Peaks')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True)

    plt.show()