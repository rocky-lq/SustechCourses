import os
import random

import torch
import torchaudio
from torchaudio import transforms

from config import TRAIN_DATA_DIR, BATCH_SIZE, GPU_NUMS, SAMPLE_RATE, DURATION, SHIFT_PCT, N_MELS, N_FFT

train_labels = os.listdir(TRAIN_DATA_DIR)

# Remove the ones that do not start with spk
train_labels = [x for x in train_labels if x.startswith('spk')]
train_labels.sort()

# Define the mapping of train_labels to index
label_to_index = {label: index for index, label in enumerate(train_labels)}
index_to_label = {index: label for label, index in label_to_index.items()}


# load an audio file, resample it to 16kHz, and return the waveform and sample rate
def open(audio_file, sample_rate):
    # if file not endwith flac or wav, return None
    if not audio_file.endswith('.flac') and not audio_file.endswith('.wav'):
        exit(0)

    sig, sr = torchaudio.load(audio_file)

    if sr == sample_rate:
        return sig, sr

    # Resample the signal
    sig = torchaudio.transforms.Resample(sr, sample_rate)(sig)
    return sig, sample_rate


def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if sig_len > max_len:
        # Truncate the signal to the given length
        start = random.randint(0, sig_len - max_len)
        sig = sig[:, start:start + max_len]

    elif sig_len < max_len:
        # Length of padding to add at the beginning and end of the signal
        pad_begin_len = random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        # Pad with 0s
        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return sig, sr


def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return sig.roll(shift_amt), sr


def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec, sr


def pre_processing_audio(audio):
    pad_audio = pad_trunc(audio, DURATION)

    shift_audio = time_shift(pad_audio, SHIFT_PCT)
    # convert to spectrogram
    spectrogram = spectro_gram(shift_audio, n_mels=N_MELS, n_fft=N_FFT, hop_len=None)

    return spectrogram


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, root, all_labels, sample_rate, type='train'):
        self.root = root
        self.type = type
        self.all_labels = all_labels
        self.set = []
        self.sr = sample_rate

        for root, dirs, files in os.walk(self.root):
            target = root.split(os.sep)[-1]
            print(target)
            if target in self.all_labels:
                for file in files:
                    audio = open(os.path.join(root, file), self.sr)

                    # convert to spectrogram
                    spectrogram, _ = pre_processing_audio(audio)

                    information = {
                        'waveform': spectrogram,
                        'target': label_to_index[target]
                    }
                    self.set.append(information)

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        return self.set[idx]


def load_training_data():
    train_set = AudioDataset(TRAIN_DATA_DIR, train_labels, SAMPLE_RATE, 'train')
    # 將train_set分为训练集和验证集
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

    batch_size = BATCH_SIZE * GPU_NUMS
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, num_workers=8)
    return train_loader, val_loader


def load_test_data():
    test_set = AudioDataset(TRAIN_DATA_DIR, train_labels, SAMPLE_RATE, 'test')
    batch_size = BATCH_SIZE * GPU_NUMS
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=8)
    return test_loader
