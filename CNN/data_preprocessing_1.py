# data_preprocessing.py
# Jason Ingersoll - Data preprocessing and input layer setup
#
# This file turns audio files into numbers the neural network can understand.
# The main idea: load a sound file -> convert it to a picture of its frequencies
# (called a spectrogram) -> flatten that picture into a list of numbers.

import csv
import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

from config import (
    SR, N_FFT, HOP_LENGTH, N_MELS, DURATION,
    NOTE_CLASSES, NUM_CLASSES, DATA_DIR, NORMALIZE_INPUT,
)


def load_audio(path, sr=SR, duration=DURATION):
    # Read the audio file and get the waveform back as a list of numbers.
    # We force everything to the same sample rate and single channel so all our inputs are consistent.
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration)

    # We then make sure every clip is exactly the same length.
        # If it's too short, add silence (zeros) at the end.
        # If it's too long, cut off the extra.
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y[:target_len]


def audio_to_mel_spectrogram(y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    # We then turn the waveform into a spectrogram (a visual representation of the sound)
    # that shows which frequencies (pitches) are present over time.
    # The "mel" part just means the frequencies are spaced out
    # the way human ears actually hear them.
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )

    # Switch to decibels (log scale) so quiet and loud parts
    # are both visible instead of loud parts drowning everything out.
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.astype(np.float32)


def spectrogram_to_vector(S):
    # The spectrogram is a 2D grid (like a small image),
    # but our network needs a single flat list of numbers.
    # So we just line up all the rows end to end.
    return S.flatten()

#this uses some of the same logic for reading the files, but calculates a global mean and standard deviation
#in order to normalize the data for CNN
def find_global_vars(root, split="train"):
    root = os.path.abspath(root)
    all_means = []
    all_std = []
    note_to_idx = {}
    for i, note in enumerate(NOTE_CLASSES):
        note_to_idx[note] = i

    # Read the CSV manifest to get the list of audio files and labels.
    # Each row has a file path and the note name.
    samples = []
    manifest_path = os.path.join(root, split + "_manifest.csv")

    if os.path.isfile(manifest_path):
        with open(manifest_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path = row["path"].strip()
                full_path = os.path.join(root, file_path)
                note = row["note"].strip()
                # Skip any notes that aren't in our list of 15
                if note not in note_to_idx:
                    continue

                # Only add the file if it actually exists on disk
                full_path = os.path.join(root, file_path)
                if os.path.isfile(full_path):
                    label = note_to_idx[note]
                    samples.append((full_path, label))

                    y = load_audio(full_path)
                    S = audio_to_mel_spectrogram(y)
                    x = spectrogram_to_vector(S)
                    mean = np.mean(x)
                    all_means.append(mean)
                    std = np.std(x) + 1e-8
                    all_std.append(std)
    global_mean = np.mean(all_means)
    global_std = np.std(all_std) + 1e-8
    return global_mean, global_std


class NoteDataset(Dataset):
    # This class lets PyTorch load our audio data during training.
    # It reads a CSV file that lists each audio file and what note it is,
    # then converts each file into a spectrogram vector when requested.

    def __init__(self, root, split="train", transform=None):
        self.root = os.path.abspath(root)
        self.transform = transform
        self.global_mean, self.global_std = find_global_vars(root, split="train")

        # The network outputs a number (0-14), so we need a way to
        # go from note names like "A" or "C#" to those numbers.
        self.note_to_idx = {}
        for i, note in enumerate(NOTE_CLASSES):
            self.note_to_idx[note] = i

        # Read the CSV manifest to get the list of audio files and labels.
        # Each row has a file path and the note name.
        self.samples = []
        manifest_path = os.path.join(self.root, split + "_manifest.csv")

        if os.path.isfile(manifest_path):
            with open(manifest_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_path = row["path"].strip()
                    note = row["note"].strip()

                    # Skip any notes that aren't in our list of 15
                    if note not in self.note_to_idx:
                        continue

                    # Only add the file if it actually exists on disk
                    full_path = os.path.join(self.root, file_path)
                    if os.path.isfile(full_path):
                        label = self.note_to_idx[note]
                        self.samples.append((full_path, label))

        print(f"Loaded {len(self.samples)} samples from {manifest_path}")
        #This initializes a cache to hold the flattened data, in order to prevent it from needing to be reloaded
        #continuously during training
        self.cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # This runs every time PyTorch asks for one sample.
        # It does the full audio -> numbers pipeline.

        if idx in self.cache:
            return self.cache[idx]
        NORMALIZE_INPUT = True
        path, label = self.samples[idx]

        # Load the sound file as a waveform
        y = load_audio(path)

        # Turn the waveform into a spectrogram (frequency picture)
        S = audio_to_mel_spectrogram(y)

        # Flatten the spectrogram into a single list of numbers
        #Update, now the spectrogram keeps its tensor shape and adds an additional dimension to it to make it
        #suitable for CNN training
        #x = spectrogram_to_vector(S)
        x = np.expand_dims(S,axis = 0)

        # Normalize: shift values so the average is 0 and scale them down.
        # Without this, the network has a harder time learning.
        if NORMALIZE_INPUT:
            #mean = np.mean(x)
            #std = np.std(x) + 1e-8  # add tiny number so we never divide by zero
            #x = (x - mean) / std
            x = (x - self.global_mean) / self.global_std

        # PyTorch needs tensors, not numpy arrays
        x = torch.from_numpy(x).float()

        if self.transform:
            x = self.transform(x)

        self.cache[idx] = (x, label)
        return x, label


def get_input_size():
    # We need to know exactly how many input numbers the network expects.
    # Easiest way: make a fake silent audio clip, run it through the
    # same spectrogram steps, and count how many numbers come out.
    dummy_len = int(SR * DURATION)
    dummy_audio = np.zeros(dummy_len, dtype=np.float32)
    S = audio_to_mel_spectrogram(dummy_audio)
    S = np.expand_dims(S, axis=0)
    return S.size


def get_input_shape():
    # Same idea as get_input_size(), but this gives the 2D spectrogram shape
    # before we flatten it into one long vector.
    dummy_len = int(SR * DURATION)
    dummy_audio = np.zeros(dummy_len, dtype=np.float32)
    S = audio_to_mel_spectrogram(dummy_audio)
    S = np.expand_dims(S, axis=0)
    return S.shape


def export_preprocessed_split(root, split="train", output_path=None):
    # This saves one whole dataset split after preprocessing so the next
    # phase of the project can load the network inputs directly.
    dataset = NoteDataset(root, split=split)

    if len(dataset) == 0:
        raise ValueError(f"No samples found for split '{split}' in {root}")

    inputs = []
    labels = []

    for i in range(len(dataset)):
        x, label = dataset[i]
        inputs.append(x)
        labels.append(label)

    # Put everything into one dictionary so it is easy to save and load later.
    exported = {
        "inputs": torch.stack(inputs),
        "labels": torch.tensor(labels, dtype=torch.long),
        "input_shape": get_input_shape(),
        "input_size": get_input_size(),
        "note_classes": NOTE_CLASSES,
        "num_classes": NUM_CLASSES,
        "split": split,
    }

    if output_path is None:
        output_path = os.path.join(os.path.abspath(root), f"{split}_preprocessed.pt")

    torch.save(exported, output_path)
    return output_path


# These give the rest of the project an easy way to import the final
# input-layer shape and size from preprocessing.
INPUT_SHAPE = get_input_shape()
INPUT_SIZE = get_input_size()
