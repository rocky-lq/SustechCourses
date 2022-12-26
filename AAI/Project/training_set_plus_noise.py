import logging
import os
from glob import glob
from pathlib import Path

import librosa
import numpy as np
import soundfile
from tqdm import tqdm

from config import SAMPLE_RATE, TRAIN_DATA_DIR, NOISE_DATA_DIR

logger = logging.getLogger(__name__)


def find_files(directory, suffix='wav'):
    return sorted(glob(directory + f'/**/*.{suffix}', recursive=True))


def train_audio_plus_noise(train_file_name, noise_file_name, sample_rate):
    train, _ = librosa.load(train_file_name, sr=sample_rate, mono=True, dtype=np.float32)
    noise, _ = librosa.load(noise_file_name, sr=sample_rate, mono=True, dtype=np.float32)

    audio_mix = np.zeros(len(train))

    if len(train) < len(noise):
        # Trim the noise audio to the same length as the training audio
        time = np.random.choice(range(0, len(noise) - len(train) + 1))
        noise = noise[time:time + len(train)]
    else:
        ...

    for i in range(len(train)):
        if i < len(noise):
            audio_mix[i] = train[i] + noise[i]
        else:
            audio_mix[i] = train[i]

    return audio_mix


def produce_train_noise_set(train_dir, noise_dir, sample_rate):
    logger.info('Producing training set with noise')
    train_files = find_files(train_dir, 'flac')
    noise_files = find_files(noise_dir, 'wav')

    # Use tqdm to show progress bar while processing the audio files
    with tqdm(train_files) as bar:
        for train_file_name in bar:
            bar.set_description(train_file_name)

            new_stem = Path(train_file_name).stem + 'noise'
            noise_file_name = np.random.choice(noise_files)

            # Mix the training and noise audio
            new_audio = train_audio_plus_noise(train_file_name, noise_file_name, sample_rate)
            new_audio_filename = Path(train_file_name).with_stem(new_stem)

            soundfile.write(new_audio_filename, new_audio, sample_rate, format='flac')


# Remove all training audio files that have 'noise' in their file names.
def clean_train_noise_set(train_dir):
    logger.info('Cleaning training set with noise')
    audio_files = find_files(train_dir, suffix='flac')
    for audio_filename in audio_files:
        if 'noise' in audio_filename:
            os.remove(audio_filename)


if __name__ == '__main__':
    # Mix training audio with noise audio and save the results to new files
    # produce_train_noise_set(TRAIN_DATA_DIR, NOISE_DATA_DIR, SAMPLE_RATE)
    # Remove the mixed audio files
    clean_train_noise_set(TRAIN_DATA_DIR)
