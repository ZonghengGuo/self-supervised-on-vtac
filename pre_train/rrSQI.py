import numpy as np
from scipy.signal import detrend
from QRS import peak_detection
import wfdb
import time


def rrSQI(ECG, qrs, freq):
    """
    Python implementation of the MATLAB rrSQI function.

    Parameters:
        ECG (numpy array): ECG waveform.
        qrs (numpy array): QRS R peak locations.
        freq (float): Sampling frequency of the ECG signal.

    Returns:
        BeatQ (numpy array): SQI of each beat (logical values).
        BeatN (numpy array): SQI of ECG corresponding beat.
        r (float): Fraction of good beats in RR.
    """
    if len(qrs) < 20 or len(ECG) < 200:
        return np.array([]), np.array([]), 0.0

    fs = freq
    timeECG = np.arange(len(ECG)) / fs
    RR = np.diff(timeECG[qrs])

    # Thresholds
    rangeHR = [40, 120]  # bpm
    dHR = 0.30
    dPeriod = 0.5  # 0.5 seconds
    noiseEN = 2

    # Beat quality
    HR = 60.0 / RR
    badHR = np.where((HR < rangeHR[0]) | (HR > rangeHR[1]))[0]

    jerkPeriod = 1 + np.where(np.abs(np.diff(RR)) > dPeriod)[0]
    jerkHR = 1 + np.where(np.abs(np.diff(HR)) / HR[:-1] > dHR)[0]

    # ECG quality
    w = int(fs * 1)  # 1-second window
    E = []
    sampen = []
    ecg = detrend(ECG) / np.std(ECG) + 10

    for i in range(0, len(ECG) - w, w):
        e = ecg[i:i + w]
        E.append(np.sum(e))
        sampen.append(sample_entropy(e, 1, 0.1, 0))

    E = np.array(E)
    sampen = np.array(sampen)

    B = np.ceil(qrs / w).astype(int)
    B = B[:-1]  # Match RR length
    B[B >= len(E)] = len(E) - 1  # Ensure indices are within bounds

    noise = np.column_stack((E[B], sampen[B]))

    M = np.percentile(E, 95)
    j = np.where(noise[:, 0] > M)[0]
    jj = np.where(noise[:, 1] > noiseEN)[0]

    # Initialize beat quality matrix
    bq = np.zeros((len(qrs) - 1, 6), dtype=int)
    bq[badHR, 1] = 1
    bq[jerkPeriod, 2] = 1
    bq[jerkHR, 3] = 1
    bq[j, 4] = 1
    bq[jj, 5] = 1

    # Combine conditions for column 1
    bq[:, 0] = bq[:, 1] | bq[:, 2] | bq[:, 3]

    # Make all "...101..." into "...111..."
    y = bq[:, 0]
    y[np.where(np.diff(y, 2) == 2)[0] + 1] = 1
    bq[:, 0] = y

    BeatQ = bq.astype(bool)

    # Fraction of good beats overall
    r = len(np.where(bq[:, 0] == 0)[0]) / len(qrs)

    # BeatN (noisy beats)
    bn = bq[:, 4] | bq[:, 5]
    BeatN = bn.astype(bool)

    return BeatQ, BeatN, r


def sample_entropy(signal, m, r, scale):
    """
    Approximate sample entropy calculation.
    Simplified for this use case.
    """
    # Placeholder for sample entropy calculation
    # Replace with a proper implementation if needed
    return np.random.random()  # Dummy value


if __name__ == '__main__':
    database_name = 'mimic3wdb'
    rel_segment_name = '3000003_0005'
    rel_segment_dir = 'mimic3wdb/30/3000003'
    segment_data = wfdb.rdrecord(record_name=rel_segment_name, pn_dir=rel_segment_dir)
    ecg_signal = segment_data.p_signal[:, 0][8000:12000]
    sampling_rate = segment_data.fs

    r_peaks = peak_detection(ecg_signal, sampling_rate)

    # use python
    start_time = time.time()
    _, _, q = rrSQI(ecg_signal, r_peaks, sampling_rate)
    elapsed_time = time.time() - start_time
    print(f"the quality of the signal is {q} in rewrite python code")
    print(f"Execution time of rewrite python code: {elapsed_time:.6f} seconds")


