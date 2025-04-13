import numpy as np
import os
import json
from tqdm import tqdm


# saving setting
ecg_segments_path = "data/mimic/wave"
qualities_path = "data/mimic/label"

pairs_save_path = "data/mimic/pair_segments"
# chunk_size = 1000
# file_counts = {i: 0 for i in range(10)}

# create dict to store segment pairs
list_pairs = {}
for i in range(10):
    list_pairs[i] = []
for key, value in list_pairs.items():
    print(f'{key}: {value}')

idx = 0

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


                    # calculate diff
                    # diff = abs(qualities[i] - qualities[j])
                    # save_index = int(diff * 10) % 10

                    # # save pairs in dict value, and key is according to diff
                    pair = [segments[i], segments[j]]
                    file_name = f"{ecg_subject_title}_{ecg_subject_name}_pair_{i}_{j}.npy"
                    file_path = os.path.join(pairs_save_path, file_name)
                    np.save(file_path, pair)

                    # list_pairs[save_index].append(pair)
                    #
                    # idx += 1
                    #
                    # if idx > chunk_size:
                    #     for key in range(10):
                    #         # create directory: 0-9
                    #         key_folder = os.path.join(pairs_save_path, str(key))
                    #         if not os.path.exists(key_folder):
                    #             os.makedirs(key_folder)
                    #
                    #         # save pairs and count using a dict(file_counts)
                    #         file_name = f"{key}_{file_counts[key]}.npy"
                    #         file_path = os.path.join(key_folder, file_name)
                    #         np.save(file_path, list_pairs[key])
                    #         print(f"Saved {len(list_pairs[key])} pairs to {file_path}")
                    #
                    #         # update the file name count and empty the pairs dict
                    #         file_counts[key] += 1
                    #         list_pairs[key] = []
                    #
                    #     # restart index
                    #     idx = 0
















