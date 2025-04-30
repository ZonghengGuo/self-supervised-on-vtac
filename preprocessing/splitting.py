import os
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

dataset_path = "data"
save_path = os.path.join(dataset_path, "out/raw")
split_path = os.path.join(dataset_path, "benchmark_data_split.csv")

# Read the benchmark data split CSV
split_df = pd.read_csv(split_path)

# Initialize lists to store data
train_samples = []
val_samples = []
test_samples = []

train_labels = []
val_labels = []
test_labels = []

# Iterate over the split dataframe
for _, row in tqdm(split_df.iterrows()):
    event = row['event']
    split = row['split']

    record_file = os.path.join(save_path, f"{event}_record.npy")
    label_file = os.path.join(save_path, f"{event}_label.npy")

    if os.path.exists(record_file) and os.path.exists(label_file):
        record = np.load(record_file)
        label = np.load(label_file)

        if split == 'train':
            train_samples.append(record)
            train_labels.append(label)
        elif split == 'val':
            val_samples.append(record)
            val_labels.append(label)
        elif split == 'test':
            test_samples.append(record)
            test_labels.append(label)

print(len(train_samples), len(val_samples), len(test_samples))

# Convert lists to tensors
train_samples = torch.tensor(np.array(train_samples))
val_samples = torch.tensor(np.array(val_samples))
test_samples = torch.tensor(np.array(test_samples))

train_labels = torch.tensor(np.array(train_labels))
val_labels = torch.tensor(np.array(val_labels))
test_labels = torch.tensor(np.array(test_labels))

# Save the datasets
output_dir = "data/out/lead_selected"
os.makedirs(output_dir, exist_ok=True)

torch.save((train_samples, train_labels), os.path.join(output_dir, "train.pt"))
torch.save((val_samples, val_labels), os.path.join(output_dir, "val.pt"))
torch.save((test_samples, test_labels), os.path.join(output_dir, "test.pt"))

print("Datasets saved successfully!")