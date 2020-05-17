import os
from tqdm import tqdm
import shutil
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
import numpy as np

if __name__ == '__main__':
    DATA_DIR = '/'.join(
        os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]) + '/datasets/google-speech-commands/'
    TARGET_NOISE_STD = 300
    NOISE_DATASET_DIR = os.path.join(DATA_DIR, 'v0.02_noise_{}_std'.format(TARGET_NOISE_STD))
    DATASET_DIR = DATA_DIR + 'v0.02'

    shutil.rmtree(NOISE_DATASET_DIR, ignore_errors=True)

    os.makedirs(NOISE_DATASET_DIR)

    for sound_dir in os.listdir(DATASET_DIR):
        noise_sound_dir = os.path.join(NOISE_DATASET_DIR, sound_dir)
        if os.path.isdir(os.path.join(DATASET_DIR, sound_dir)):
            os.makedirs(noise_sound_dir)

        if os.path.isfile(os.path.join(DATASET_DIR, sound_dir)):
            shutil.copyfile(os.path.join(DATASET_DIR, sound_dir), os.path.join(NOISE_DATASET_DIR, sound_dir))
            continue
        for sound in tqdm(os.listdir(os.path.join(DATASET_DIR, sound_dir))):
            try:
                rate, pure = wav_read(os.path.join(DATASET_DIR, sound_dir, sound))
                noise = np.random.normal(0, TARGET_NOISE_STD, pure.shape)
                signal = pure + noise
                signal = signal.astype(np.int16)
                wav_write(os.path.join(NOISE_DATASET_DIR, sound_dir, sound), rate, signal)
            except:
                print('Exception with {} file'.format(sound))
                shutil.copyfile(os.path.join(DATASET_DIR, sound_dir, sound),
                                os.path.join(NOISE_DATASET_DIR, sound_dir, sound))
