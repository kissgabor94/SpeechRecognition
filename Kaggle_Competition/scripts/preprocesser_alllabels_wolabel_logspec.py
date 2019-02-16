#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 23:17:59 2017

@author: gabor
"""

import random, os
import numpy as np
from sklearn import preprocessing
import librosa
import gc
from scipy.fftpack import fft
from scipy import signal
from sklearn.externals import joblib
import pandas as pd
#%%

labels = {"yes": 0, "no": 1, "up": 2, "down": 3, "left": 4, "right": 5, "on": 6, "off": 7, "stop": 8, "go": 9, "zero": 10, "one": 11, "two": 12, "three": 13, "four": 14, "five": 15, "six": 16, "seven": 17, "eight": 18, "nine": 19, "bed": 20, "bird": 21, "cat": 22, "dog": 23, "happy": 24, "house": 25, "marvin": 26, "sheila": 27, "tree": 28, "wow": 29, "silence": 30}
#%%
L=16000
def custom_fft(y, fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    # FFT is simmetrical, so we take just the first half
    # FFT is also complex, to we take just the real part (abs)
    vals = 2.0/N * np.abs(yf[0:N//2])
    return xf, vals

def log_specgram(audio, sample_rate, window_size=20,
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)
def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=20):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]
#%%
train = []
root="/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/train/audio"
dirs = os.listdir(root)# list dirs
for folder in dirs: # for every directory
    if folder != "_background_noise_": # leave noise out of this
        print(folder)
        files = os.listdir(root + "/" + folder) # every file from the folder
        index = 0 #index of the label array
        if labels.__contains__(folder): # word is in the dictionary
            index = labels[folder]
        else: #unknown word
            index = 11
        
        for file in files: # for every file in the folder
            substr = file[0:8]
            if substr > "3ab9ba07":
                filename = root + "/" + folder + "/" + file
                y, sr = librosa.load(filename) #read in the file
                y = pad_audio(y)
                if (len(y)>L):
                    y = y[0:15999]
                _, _, inputs = log_specgram(y, sample_rate=sr)
                #inputs = reshape(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=8000),size=[40,101])
                """if y.shape[0] != sr:
                    y = np.append(y, np.zeros((sr - y.shape[0], )))
                inputs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=8000)
                mfcc_delta = librosa.feature.delta(inputs)
                mfcc_delta2 = librosa.feature.delta(inputs, order=2)
                inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)"""
                outputs = np.zeros(31)
                outputs[index]=1
                temp = []
                temp.append(inputs)
                temp.append(outputs)
                train.append(temp)
            """if substr <= "1fd85ee4":
                valid.append(temp) 
            elif substr <= "3ab9ba07":
                test.append(temp)
            else:
                train.append(temp)"""
        # foreach files
    #if not background noise
#foreach dirs
#%%
dirs = None 
#%% create silence clips
def onesecs(clip):
    l=len(clip)
    done = True
    i=0
    ret=[]
    while(done):
        if ((i+16000)<l):       
            tmp = clip[i:i+16000]
            ret.append(tmp)
            i+=8000
        else:
            done = False
    return ret

base = "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/train/audio/_background_noise_/"
something = [base + "doing_the_dishes.wav", base + "dude_miaowing.wav", base + "exercise_bike.wav", base + "pink_noise.wav", base + "running_tap.wav", base + "white_noise.wav"]
for i in range(6):
    if i<4:
        y, sr = librosa.load(something[i])
        if sr>16000:
            y = signal.resample(y, int(16000/sr*len(y)))
        clips = onesecs(y)
        for f in clips:
            y = pad_audio(y)
            if (len(y)>L):
                y = y[0:15999]
            _, _, inputs = log_specgram(y, sample_rate=sr)
            """inputs = reshape(librosa.feature.mfcc(y=f, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=8000),size=[40,101])
            mfcc_delta = librosa.feature.delta(inputs)
            mfcc_delta2 = librosa.feature.delta(inputs, order=2)
            inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)"""
            outputs = np.zeros(31)
            outputs[30]=1
            temp = []
            temp.append(inputs)
            temp.append(outputs)
            train.append(temp)

#%% standardization 
random.shuffle(train) #shuffle the dataset
shape = np.shape(train[0][0]) #original shape
X_train = np.zeros((len(train), shape[0], shape[1], 1)) #pre-allocate np matrices for the data
Y_train = np.zeros((len(train), 31)) #and the labels
temp_train = []
# Here I reshape the inputs to 1D arrays (this is necessary for the standardization)
for item in train:
    temp_train.append(np.reshape(item[0], -1, order="F")) #do not change the order

scaler = preprocessing.StandardScaler().fit(temp_train) #calculate the scaler from the train data
#scale the train, validation and test datasets
temp_train = scaler.transform(temp_train)

#Here I reshape the scaled data to the original shape
for i in range(len(train)):
    X_train[i,:,:,:]=np.atleast_3d(np.reshape(temp_train[i,:], shape, order="F"))
    Y_train[i,:]=train[i][1]
temp_train = None
train = None
gc.collect()
path="/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/"
np.savez_compressed(path+"train_specgram_alllabels_wolabel.npz", inputs = X_train, outputs=Y_train)
#%% save scaler
joblib.dump(scaler, "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/scaler_specgram_alllabels_wolabel.pkl")
