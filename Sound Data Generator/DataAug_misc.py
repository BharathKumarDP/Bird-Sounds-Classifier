import numpy as np
import random

np.random.seed(42)

import os
import glob

from librosa import load, stft, power_to_db
from librosa.feature import melspectrogram, mfcc

n_fft = 512
hop_length = 128
sr=22050 
window='hann'
n_mfcc = 13
duration = 3

sample = np.zeros((sr * duration,))

def time_shift_spectrogram(spectrogram):
    
    nb_cols = spectrogram.shape[1]
    nb_shifts = np.random.randint(0, nb_cols)

    return np.roll(spectrogram, nb_shifts, axis=1)

def pitch_shift_spectrogram(spectrogram):

    nb_cols = spectrogram.shape[0]
    max_shifts = nb_cols//20
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(spectrogram, nb_shifts, axis=0)

def same_class_augmentation(wave, class_dir):

    sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    aug_sig_path = random.choice(sig_paths)
    aug_sig, sr = load(aug_sig_path)
    
    alpha = np.random.rand()
    wave = (1.0-alpha)*wave + alpha*aug_sig
    
    return wave

def noise_augmentation(wave, noise_files):

    nb_noise_segments = 3
    aug_noise_files = []
    for i in range(nb_noise_segments):
        aug_noise_files.append(random.choice(noise_files))

    dampening_factor = 0.4
    for aug_noise_path in aug_noise_files:
        aug_noise, sr = load(aug_noise_path)
        wave = wave + aug_noise*dampening_factor
    
    return wave

def wav_to_mfcc(x):

    mfcc_wav = mfcc(x, n_mfcc=13, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mfcc_wav = mfcc_wav.reshape(mfcc_wav.shape[0], mfcc_wav.shape[1], 1)
    
    return mfcc_wav

def wav_to_mel(x):

    mel_spec = melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, window= window)
    mel_spec = mel_spec.reshape(mel_spec.shape[0], mel_spec.shape[1], 1)
    
    return mel_spec

def wav_to_power(x):
    
    power_spec = np.abs(stft(x, n_fft=n_fft, hop_length=hop_length, window=window))**2
    power_spec = power_spec.reshape(power_spec.shape[0], power_spec.shape[1], 1)
    return power_spec

def wav_to_mel_util(x):
    mel_spec = melspectrogram(x, sr=sr, n_fft=n_fft, hop_length=hop_length, window= window)
    return mel_spec

def mel_to_mfcc(x):
    mfcc_wav = mfcc(S = power_to_db(x), n_mfcc=13, sr=sr, n_fft = n_fft, hop_length=hop_length)
    mfcc_wav = mfcc_wav.reshape(mfcc_wav.shape[0], mfcc_wav.shape[1], 1)
    return mfcc_wav

def target_size_calc(output_mode):
    if(output_mode == 'mfcc'):
        target_size = (mfcc(sample, n_mfcc=13, sr=sr, n_fft=n_fft,hop_length=hop_length)).shape
     
    elif(output_mode == 'mel_spec'):
        target_size = (melspectrogram(sample, sr=sr, n_fft=n_fft, hop_length=hop_length, window= window)).shape
    
    else:
        target_size = (np.abs(stft(sample, n_fft=n_fft, hop_length=hop_length, window=window))**2).shape
    
    targetSize = (target_size[0], target_size[1], 1)
    
    return targetSize
    
