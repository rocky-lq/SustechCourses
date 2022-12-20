import os

SAMPLE_RATE = 16000

BATCH_SIZE = 32
GPU_NUMS = 6

WORKING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'LibriSpeech-SI')

TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'train')
NOISE_DATA_DIR = os.path.join(DATA_DIR, 'noise')
TEST_DATA_DIR = os.path.join(DATA_DIR, 'test')
TEST_NOISY_DATA_DIR = os.path.join(DATA_DIR, 'test_noisy')
