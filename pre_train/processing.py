from pathlib import PurePosixPath
import wfdb
from tqdm import tqdm
import posixpath
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
import os
from rrSQI import rrSQI
from QRS import peak_detection
import requests
from ppg_SQI import ppg_SQI
import pandas as pd

# get  waveform, classify it into 5 classes
# 5min = 15 pairs, I need 10k pairs, in total 3500 min


# Setting parameters
database_name = "mimic3wdb"
required_sigs = ["PLETH", "II", "V"]  # we select the longest lead: 'II'
shortest_minutes = 5
req_seg_duration = 5 * 60
seg_save_path = "data/mimic/wave"
qua_save_path = "data/mimic/label"
slide_segment_time = 30  # ~ seconds window size
nan_limit = 0.1
original_fs = 125  # mimic3wfdb ecg fs
required_minutes = 3500


# target_fs = setting["target_fs"]  # downsample fs


# Preprocessing function
def is_nan_ratio_exceed_any(sig, threshold, fs, segment_time):
    nan_ratios = np.isnan(sig).sum(axis=1) / (fs * segment_time)
    return np.any(nan_ratios > threshold)


# Downsample signals
def downsample(signal_data, fs_orig, fs_new):
    ratio = Fraction(fs_new, fs_orig).limit_denominator()
    up = ratio.numerator
    down = ratio.denominator
    return resample_poly(signal_data, up, down)


# Min-max normalization
def min_max_normalize(signal, feature_range=(-1, 1)):
    min_val, max_val = feature_range
    signal = np.asarray(signal)

    # Apply min-max normalization to each channel independently
    min_signal = np.min(signal, axis=1, keepdims=True)  # shape (4, 1)
    max_signal = np.max(signal, axis=1, keepdims=True)  # shape (4, 1)

    # Avoid division by zero if all values in a channel are the same
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized_signal = (signal - min_signal) / (max_signal - min_signal)
        normalized_signal = normalized_signal * (max_val - min_val) + min_val

    # If max_signal equals min_signal (i.e., constant signal), set to zero or min_val
    normalized_signal[np.isnan(normalized_signal)] = min_val

    return normalized_signal


def set_nan_to_zero(sig):
    zero_segment = np.nan_to_num(sig, nan=0.0)
    return zero_segment


def interpolate_nan_multichannel(sig):
    # sig: shape (channels, time)
    interpolated = []
    for channel in sig:
        interpolated_channel = pd.Series(channel).interpolate(method='linear', limit_direction='both').to_numpy()
        interpolated.append(interpolated_channel)
    return np.array(interpolated)


def is_any_constant_signal(slide_segment):
    return np.any(np.all(slide_segment == slide_segment[:, [0]], axis=1))


def scale_ppg_score(qua_ppg):
    qua_ppg_scaled = (qua_ppg - 0.5) / 0.3
    qua_ppg_scaled = max(0, min(qua_ppg_scaled, 1))
    return qua_ppg_scaled


# Select the suitable segments with 'II' lead and time length > f{"shortest_minutes"}
# index record data path
def process_record(record):
    global required_minutes  # 显式声明我们要使用的是全局变量
    record_name = record.name
    record_path = posixpath.join(database_name, record.parent, record_name)

    # Skip the empty record or skip network disconnection or anyother read error
    try:
        record_data = wfdb.rdheader(record_name, pn_dir=record_path, rd_segments=True)
    except FileNotFoundError:
        print(f"Record {record_name} not found, skipping...")
        return
    except requests.exceptions.RequestException as e:
        print(f"Network error while accessing {record_name}, skipping... Error: {e}")
        return
    except Exception as e:
        print(f"Error processing {record_name}, skipping... Error: {e}")
        return

    # index segments according to the record data path
    segments = record_data.seg_name
    for segment in segments:
        if segment == "~":
            continue
        segment_metadata = wfdb.rdheader(record_name=segment, pn_dir=record_path)

        print('-------------------')
        print(f"Start preprocessing {record_path}/{segment}")

        # Check if the segments include required lead
        sigs_leads = segment_metadata.sig_name

        if len(sigs_leads) < 2:
            print("Not enough channels, skip..")
            continue

        if not all(x in sigs_leads for x in required_sigs):
            print(f'{sigs_leads} is missing signal of II, PLETH')
            continue

        # check if the segments is longer than f{shortest_minutes}
        seg_length = segment_metadata.sig_len / (segment_metadata.fs)
        if seg_length < req_seg_duration:
            print(f' (too short at {seg_length / 60:.1f} mins)')
            continue

        print(f"Have the {sigs_leads}..........")

        matching_seg = posixpath.join(record_path, segment)  # "mimic3wdb/32/3213671/3213671_0002"

        # required_minutes -= seg_length / 60
        # print(f' (met requirements), left {required_minutes} needs to be stored')
        # if required_minutes < 0:
        #     print("Collection completed!!!!")
        #     break

        # segment every signal into 30s slides
        seg_sig = wfdb.rdrecord(segment, pn_dir=record_path)
        sig_ppg_index = seg_sig.sig_name.index('PLETH')
        sig_ii_index = seg_sig.sig_name.index('II')

        sig_ppg = seg_sig.p_signal[:, sig_ppg_index]
        sig_ii = seg_sig.p_signal[:, sig_ii_index]

        multi_channel_signal = np.stack((sig_ppg, sig_ii), axis=0)

        # setting
        slide_segment_length = slide_segment_time * original_fs
        slide_segments = []
        qua_labels = []

        # divide into 30 sec, and discard the last slide (<30s)
        for i, start in enumerate(range(0, len(sig_ppg) - slide_segment_length + 1, slide_segment_length)):
            end = start + slide_segment_length
            slide_segment = multi_channel_signal[:, start:end]

            # check if too much nan value
            if is_nan_ratio_exceed_any(slide_segment, nan_limit, original_fs, slide_segment_time):
                print(f"too much missing value, nan ratio is {((np.isnan(slide_segment).sum() / 3750) * 100):.2f}%")
                continue

            # check if the signal is stable
            if is_any_constant_signal(slide_segment):
                print(f"the sequence is stable, not a signal")
                continue

            # interpolate
            slide_segment = interpolate_nan_multichannel(slide_segment)
            print("set nan value to zero and normalize signal")

            lead_ppg_segments = slide_segment[0, :]
            lead_ii_segments = slide_segment[1, :]

            # ECG quality assessment
            try:
                peaks = peak_detection(lead_ii_segments, original_fs)
                print("find peaks")
            except ValueError as e:
                print(f"Warning: {e}, skipping this segment.")
                continue

            _, _, qua_ii = rrSQI(lead_ii_segments, peaks, original_fs)

            # PPG quality assessment
            qua_ppg = ppg_SQI(lead_ppg_segments, original_fs)
            qua_ppg = scale_ppg_score(qua_ppg)

            qua = (qua_ii + qua_ppg) / 2

            if qua >= 0.9:
                label = "Excellent"
            elif 0.9 < qua <= 0.7:
                label = "Good"
            elif 0.7 < qua <= 0.5:
                label = "Acceptable"
            elif 0.5 < qua <= 0.3:
                label = "Poor"
            else:
                label = "Bad"

            # Todo: give classification to qua
            qua_labels.append(label)
            print(f"The quality in {str(record.name)}.npy_{i} is: {qua}")

            # downsample to 40Hz
            # slide_segment = downsample(slide_segment, original_fs, target_fs)

            slide_segment = min_max_normalize(slide_segment)

            slide_segments.append(slide_segment)

        # save the segments and qualities list
        segment_save_path = seg_save_path + '/' + str(record.parent) + '/' + str(record.name) + '/' + segment
        os.makedirs(os.path.dirname(segment_save_path), exist_ok=True)

        quality_save_path = qua_save_path + '/' + str(record.parent) + '/' + str(record.name) + '/' + segment
        os.makedirs(os.path.dirname(quality_save_path), exist_ok=True)

        np.save(segment_save_path, slide_segments)
        try:
            np.save(quality_save_path, qua_labels)
        except ValueError as e:
            print(f"Skip wrong dimension of qua_labels in {record_path}/{segment}")
            continue

        print(f"save segments into: {segment_save_path}.npy and qualities into {quality_save_path}.npy")


async def async_process_records(records):
    print(f"Using {os.cpu_count()} cpu cores for synchronous programming and multi-thread pool processing")
    with ProcessPoolExecutor(max_workers=1) as pool:
        loop = asyncio.get_event_loop()
        tasks = []
        with tqdm(total=len(records), desc="Processing records") as pbar:
            for record in records:
                task = loop.run_in_executor(pool, process_record, record)
                task.add_done_callback(lambda _: pbar.update())
                tasks.append(task)

            await asyncio.gather(*tasks)


async def main(records):
    await async_process_records(records)


if __name__ == '__main__':
    min_records_to_load = 10000
    max_records_to_load = 67830
    # total: 67830

    # Get the database and records
    subjects = wfdb.get_record_list(database_name)
    print(f"The '{database_name}' database contains data from {len(subjects)} subjects")
    all_records = wfdb.get_record_list(database_name)
    sequent_records = all_records[min_records_to_load: max_records_to_load]
    records = [PurePosixPath(record) for record in sequent_records]
    print(f"Loaded {len(records)} records from the '{database_name}' database.")

    for record in records:
        process_record(record)