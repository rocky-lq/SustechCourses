import os
import platform

import torch

SAMPLE_RATE = 16000

BATCH_SIZE = 32
NUM_CLASSES = 250

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'LibriSpeech-SI')

TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
NOISE_DATA_DIR = os.path.join(DATA_DIR, 'noise')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
TEST_NOISY_DATA_DIR = os.path.join(DATA_DIR, 'test-noisy')
TRAIN_NOISY_DATA_DIR = os.path.join(DATA_DIR, 'train-noisy')

# pad the audio
DURATION = 18750

# time shift
SHIFT_PCT = 0.4

# convert to spectrogram
N_MELS = 64
N_FFT = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if platform.system() == 'Windows':
    device_ids = [0]  # test platform, just have one GPU
else:
    device_ids = [0, 1, 2, 3, 4, 5]  # Server platform, have 6 GPUs

GPU_NUMS = len(device_ids)
