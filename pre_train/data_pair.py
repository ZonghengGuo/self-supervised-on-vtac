import numpy as np
import os
import json
from tqdm import tqdm


# saving setting
ecg_segments_path = "data/mimic/wave"
qualities_path = "data/mimic/label"

pairs_save_path = "data/mimic/pair_segments"

quality_rank = {
    "Excellent": 0,
    "Good": 1,
    "Acceptable": 2,
    "Poor": 3,
    "Bad": 4
}

def compare_quality(q1, q2):
    return quality_rank[q1] < quality_rank[q2]


for ecg_subject_title in os.listdir(ecg_segments_path):
    ecg_subject_title_path = os.path.join(ecg_segments_path, ecg_subject_title)
    for ecg_subject_name in os.listdir(ecg_subject_title_path):
        ecg_subject_path = os.path.join(ecg_subject_title_path, ecg_subject_name)

        # display process bar
        ecg_segments_list = os.listdir(ecg_subject_path)
        for ecg_segments_name in tqdm(ecg_segments_list, desc=f"Processing {ecg_subject_name}"):

            # read segments.npy
            ecg_slide_segments_path = os.path.join(ecg_subject_path, ecg_segments_name)
            segments = np.load(ecg_slide_segments_path)

            # read corresponding quality label
            quality_path = os.path.join(qualities_path, ecg_subject_title, ecg_subject_name, ecg_segments_name)
            qualities = np.load(quality_path)

            # Find surrounding 5min segments
            n = len(segments)
            for i in range(n):
                for j in range(i + 1, min(i + 10, n)):
                    # if two samples qualities are the same, skip this pair
                    if qualities[i] == qualities[j]:
                        continue

                    if segments[i].size == 0 or segments[j].size == 0:
                        print("There is no value, skip....")
                        continue

                    if qualities[i] == 0 or qualities[j] == 0:
                        continue

                    # # save pairs in dict value, and key is according to diff

                    # if ith-quality is better than j-th quality, then save [segments[i], segments[j]]
                    # Reversly, if j-th is better, save [segments[j], segments[i]]
                    # the first segment is relatively good signal, and second is bad.
                    if compare_quality(qualities[i], qualities[j]):
                        pair = [segments[i], segments[j]]
                    else:
                        pair = [segments[j], segments[i]]

                    file_name = f"{ecg_subject_title}_{ecg_subject_name}_pair_{i}_{j}.npy"
                    file_path = os.path.join(pairs_save_path, file_name)
                    np.save(file_path, pair)
