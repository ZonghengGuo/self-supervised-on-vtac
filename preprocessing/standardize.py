"""
Z-score normalization and per-sample clipping
"""
import torch
from rich import print
from rich.progress import track
from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np

def plot_four_channels(original, standardized):
    plt.figure(figsize=(12, 16))

    for i in range(4):
        plt.subplot(4, 2, 2*i + 1)
        plt.plot(original[i], label=f'Original Channel {i}')
        plt.title(f'Original Channel {i}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(4, 2, 2*i + 2)
        plt.plot(standardized[i], label=f'Standardized Channel {i}')
        plt.title(f'Standardized Channel {i}')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()

    plt.tight_layout()
    plt.show()

def compute_mean_std():
    samples, _ = torch.load("../../database/vtac/out/filtered/train-filtered.pt", weights_only=False)
    samples = samples[:, :, 72500:75000]
    mu = []
    sigma = []
    # for each signal
    for i in range(samples.shape[1]):
        vals = []
        for x in range(len(samples)):
            sample = samples[x, i]
            # if current signal is available
            if sample.sum() != 0.0:
                vals.append(sample)

        vals = torch.cat(vals)
        mu.append(vals.mean())
        sigma.append(vals.std())
    return mu, sigma


def create_splits(mu, sigma):
    """
    Use the population mean and standard deviation to normalize each sample
    Saves the output to out/population-norm/{split}.pt

    Args:
        mu: list of population means for each channel
        sigma: list of population standard deviations for each channel

    Returns:
        None
    """
    for split in ["train", "val", "test"]:
        samples, ys = torch.load(f"../../database/vtac/out/filtered/{split}-filtered.pt", weights_only=False)
        num_channels = samples.shape[1]
        for i in range(num_channels):
            mu_i = mu[i]
            sigma_i = sigma[i]
            for x in track(
                range(len(samples)), description="Normalizing...", transient=True
            ):
                if samples[x, i].sum() != 0.0:
                    samples[x, i] = (samples[x, i] - mu_i) / sigma_i

        samples = samples.float()
        torch.save((samples, ys), f"../../database/vtac/out/population-norm/{split}.pt")


def create_individual_splits():
    """
    Normalize each sample individually
    Saves the output to data/out/sample-norm/{split}.pt
    """
    for split in ["train", "val", "test"]:
        samples, ys = torch.load(f"../../database/vtac/out/filtered/{split}-filtered.pt", weights_only=False)

        oringinal_samples = samples.clone()

        num_channels = samples.shape[1]
        for i in range(num_channels):
            for x in track(
                range(len(samples)), description="Normalizing...", transient=True
            ):
                if samples[x, i].sum() != 0.0:
                    mu_i = samples[x, i, 72500:75000].mean()
                    sigma_i = samples[x, i, 72500:75000].std()
                    samples[x, i] = (samples[x, i] - mu_i) / sigma_i

        samples = samples.float()

        for i in range(samples.shape[0]):
            plot_four_channels(oringinal_samples[i, :, 0:15000], samples[i, :, 0:15000])


        torch.save((samples, ys), f"../../database/vtac/out/sample-norm/{split}.pt")


if __name__ == "__main__":
    print("Use population normalization or per-sample normalization?")
    print("1. Population normalization")
    print("2. Per-sample normalization")
    choice = input("Enter your choice (1 or 2):")
    if choice == "1":
        mu, sigma = compute_mean_std()
        create_splits(mu, sigma)
    elif choice == "2":
        create_individual_splits()
    else:
        print("Invalid choice. Exiting...")
        exit(1)
