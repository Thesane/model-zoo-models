import os
from tqdm import tqdm
import shutil
from scipy.io.wavfile import read as wav_read
from scipy.io.wavfile import write as wav_write
import numpy as np
from random import shuffle


def signaltonoise(a, axis, ddof):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


def get_stats(row, name):
    _mean = np.round(np.mean(row), 3)
    _max = np.round(np.max(row), 3)
    _std = np.round(np.std(row), 3)
    print('For {} mean: {} max: {} std: {}'.format(name, _mean, _max, _std))

# EXAMPLE OF HAR WORDS: tree,three,go,dog,no,on,bed,seven,left,zero,stop

if __name__ == '__main__':
    DATA_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]) + '/workspace/datasets/google-speech-commands/'
    TARGET_NOISE_STD = 500
    NOISE_DATASET_DIR = os.path.join(DATA_DIR, 'eval_noise_{}_std'.format(TARGET_NOISE_STD))
    DATASET_DIR = DATA_DIR + 'eval'
    EXAMPLE_OF_NOISE = os.path.join(DATASET_DIR, 'backward', '0cb74144_nohash_0.wav')


    def get_random_word(except_word=set(), scale=1.0):
        _dirs = os.listdir(DATASET_DIR)
        shuffle(_dirs)
        for sound_dir in _dirs:

            if os.path.isdir(os.path.join(DATA_DIR + 'train', sound_dir)) and sound_dir not in except_word:
                all_sounds = os.listdir(os.path.join(DATA_DIR + 'train', sound_dir))
                shuffle(all_sounds)
                if len(all_sounds) > 2:
                    example_sound_sample = all_sounds[np.random.randint(0, 30)]
                else:
                    example_sound_sample = all_sounds[0]
                rate, example_sound = wav_read(os.path.join(DATA_DIR + 'train', sound_dir, example_sound_sample))

                sliced_example_sound = example_sound

                sliced_example_sound = sliced_example_sound * scale

                sliced_example_sound = sliced_example_sound.astype(np.int16)
                return sliced_example_sound





    rate, example_noise = wav_read(EXAMPLE_OF_NOISE)
    noise_size = example_noise.size
    example_noise = np.repeat(example_noise, 10)
    example_noise *= 3

    # .std(pure), np.std(noise_sllice), np.std(pure + noise_sllice)

    shutil.rmtree(NOISE_DATASET_DIR, ignore_errors=True)

    os.makedirs(NOISE_DATASET_DIR)

    for sound_dir in os.listdir(DATASET_DIR):
        std_pure = []
        std_noise_slice = []
        std_noised_signal = []
        snr = []

        noise_sound_dir = os.path.join(NOISE_DATASET_DIR, sound_dir)
        if os.path.isdir(os.path.join(DATASET_DIR, sound_dir)):
            os.makedirs(noise_sound_dir)

        if os.path.isfile(os.path.join(DATASET_DIR, sound_dir)):
            shutil.copyfile(os.path.join(DATASET_DIR, sound_dir), os.path.join(NOISE_DATASET_DIR, sound_dir))
            continue


        for num, sound in tqdm(enumerate(os.listdir(os.path.join(DATASET_DIR, sound_dir)))):
            if sound in ['_silence_', '.DS_Store']:
                continue
            try:
                rate, pure = wav_read(os.path.join(DATASET_DIR, sound_dir, sound))
            except:
                import pdb
                pdb.set_trace()
            noise = np.random.normal(0, TARGET_NOISE_STD, pure.shape)
            shift = np.random.randint(int(noise_size * 0.1), int(noise_size * 0.9))
            # noise_slice = example_noise[shift: shift + pure.size]

            noise_slice = get_random_word(('_background_noise_', sound_dir), scale=0.6)[:pure.size]
            noise_slice = np.pad(noise_slice, int((pure.size - noise_slice.size) / 2) + 2)
            noise_slice = noise_slice[:pure.size][::-1]

            coef = max(np.percentile(pure, 95) / np.percentile(noise_slice, 83), 1)
            try:
                coef = int(min(coef, 2))
            except:
                coef = 1
            noise_slice *= coef

            noised_signal = pure + noise_slice

            std_pure.append(np.std(pure).round(3))
            std_noise_slice.append(np.std(noise_slice).round(3))
            std_noised_signal.append(np.std(noised_signal).round(3))
            snr.append(np.sqrt(np.sum(np.power(pure, 2))) / np.sqrt(np.sum(np.power(noise_slice, 2))))

            noised_signal = noised_signal.astype(np.int16)
            wav_write(os.path.join(NOISE_DATASET_DIR, sound_dir, sound), rate, noised_signal)
        snr = np.nan_to_num(np.array(snr))
        snr = snr[snr != 0]

        print('For sounds : {}'.format(sound_dir))
        get_stats(std_pure, 'std_pure')
        get_stats(std_noise_slice, 'std_noise_slice')
        get_stats(std_noised_signal, 'std_noised_signal')
        get_stats(snr, 'snr')
        print('records count with snr below one: {} [ {}%]'.format(snr[snr < 1].size, round(snr[snr < 1].size / snr.size, 2)))
        print('\n\n\n')
            # except Exception as e:
            #     print('Exception with {} file: {}'.format(sound, e))
            #     shutil.copyfile(os.path.join(DATASET_DIR, sound_dir, sound),
            #                     os.path.join(NOISE_DATASET_DIR, sound_dir, sound))
