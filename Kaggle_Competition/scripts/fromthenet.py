#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:06:43 2017

@author: gabor
"""

import os
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization, GRU, Convolution1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import keras
#%%
L = 16000
legal_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]

#src folders
root_path = '/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition'
out_path = '.'
model_path = '.'
train_data_path = os.path.join(root_path,  'train', 'audio/')
test_data_path = os.path.join(root_path,  'test', 'audio/')

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

def list_wavs_fname(dirpath, ext='wav'):
    print(dirpath)
    fpaths = glob("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/train/audio/*/*.wav")
    pat = '.+/(\w+)/\w+\.' + ext + '$'
    labels = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            labels.append(r.group(1))
    pat = '.+/(\w+\.' + ext + ')$'
    fnames = []
    for fpath in fpaths:
        r = re.match(pat, fpath)
        if r:
            fnames.append(r.group(1))
    return labels, fnames

def pad_audio(samples):
    if len(samples) >= L: return samples
    else: return np.pad(samples, pad_width=(L - len(samples), 0), mode='constant', constant_values=(0, 0))

def chop_audio(samples, L=16000, num=500):
    for i in range(num):
        beg = np.random.randint(0, len(samples) - L)
        yield samples[beg: beg + L]

def label_transform(labels):
    nlabels = []
    for label in labels:
        if label == '_background_noise_':
            nlabels.append('silence')
        elif label not in legal_labels:
            nlabels.append('unknown')
        else:
            nlabels.append(label)
    return pd.get_dummies(pd.Series(nlabels))
#%%
"""fpaths = glob("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/train/audio/*/*.wav")
pat = '.+/(\w+)/\w+\.' + ext + '$'
r=re.match(pat, fpaths[0])
print(r)"""
labels, fnames = list_wavs_fname(train_data_path)
#%%
new_sample_rate = 16000
y_train = []
x_train = []

for label, fname in zip(labels, fnames):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    if len(samples) > 16000:
        n_samples = chop_audio(samples)
    else: n_samples = [samples]
    for samples in n_samples:
        #resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        _, _, specgram = log_specgram(samples, sample_rate=sample_rate)
        y_train.append(label)
        x_train.append(specgram)
x_train = np.array(x_train)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
y_train = label_transform(y_train)
label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)
del labels, fnames
gc.collect()    
#%%  
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2018)
#%%
from keras.callbacks import ModelCheckpoint, EarlyStopping
model_path = "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/models"
#%%
patience=15
early_stopping=EarlyStopping(patience=patience, verbose=1)
checkpointer=ModelCheckpoint(os.path.join(model_path, 'cnn18.model'), save_best_only=True, verbose=1)
input_shape = (99, 161, 1)
nclass = 31
inp = Input(shape=input_shape)
norm_inp = BatchNormalization()(inp)
img_1 = Convolution2D(22, kernel_size=2, activation=activations.relu, padding = 'same')(norm_inp)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution2D(22, kernel_size=2, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Convolution2D(35, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution2D(35, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Convolution2D(70, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution2D(70, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Convolution2D(130, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution2D(130, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Convolution2D(160, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution2D(160, kernel_size=3, activation=activations.relu, padding = 'same')(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Flatten()(img_1)
"""img_1 = Convolution1D(30, kernel_size=2, activation = activations.relu)(img_1)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution1D(55, kernel_size=3, activation = activations.relu)(norm_inp)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution1D(80, kernel_size=3, activation = activations.relu)(norm_inp)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution1D(120, kernel_size=3, activation = activations.relu)(norm_inp)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution1D(150, kernel_size=3, activation = activations.relu)(norm_inp)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.25)(img_1)"""
"""img_1 = Convolution1D(30, kernel_size=2, activation = activations.relu)(norm_inp)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = Convolution1D(120, kernel_size=3, activation = activations.relu)(norm_inp)
img_1 = MaxPooling1D(pool_size=2)(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = GRU(350, return_sequences=True)(img_1)
img_1 = Dropout(rate=0.25)(img_1)
img_1 = GRU(150, return_sequences=False)(img_1)
img_1 = Dropout(rate=0.25)(img_1)"""

"""dense_1 = BatchNormalization()(Dense(200, activation=activations.relu)(img_1))
dense_1 = Dropout(rate=0.25)(dense_1)
dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
dense_1 = Dropout(rate=0.25)(dense_1)"""
dense_1 = Dense(200, activation=activations.relu)(img_1)
dense_1 = Dropout(rate=0.25)(dense_1)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(128, activation=activations.relu)(dense_1)
dense_1 = Dropout(rate=0.25)(dense_1)
dense_1 = BatchNormalization()(dense_1)
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam()

model.compile(optimizer=opt, loss=losses.binary_crossentropy)                                                                                                                                                                                                                                                                                               
model.summary()
#%%
model.fit(x_train, y_train, batch_size=64, validation_data=(x_valid, y_valid), epochs=1000, callbacks=[checkpointer, early_stopping],shuffle=True, verbose=1)
#%%
"""from keras.models import load_model
model = load_model(os.path.join(model_path, 'cnn16.model'))
model.summary()"""
#%%
from keras.models import load_model
model = load_model(os.path.join(model_path, 'cnn18.model'))
out_path = "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition"
new_sample_rate = 16000
def test_data_generator(batch=16):
    fpaths = glob(os.path.join(test_data_path, '*wav'))
    i = 0
    for path in fpaths:
        if i == 0:
            imgs = []
            fnames = []
        i += 1
        rate, samples = wavfile.read(path)
        samples = pad_audio(samples)
        #resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
        _, _, specgram = log_specgram(samples, sample_rate=rate)
        imgs.append(specgram)
        fnames.append(path.split('\\')[-1])
        if i == batch:
            i = 0
            imgs = np.array(imgs)
            imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
            yield fnames, imgs
    if i < batch:
        imgs = np.array(imgs)
        imgs = imgs.reshape(tuple(list(imgs.shape) + [1]))
        yield fnames, imgs
    raise StopIteration()
    
#exit() #delete this
#del x_train, y_train
gc.collect()

index = []
results = []
hol = 0
for fnames, imgs in test_data_generator(batch=64):
    if hol % 3200 == 0:
        print(hol)
    predicts = model.predict(imgs)
    predicts = np.argmax(predicts, axis=1)
    predicts = [label_index[p] for p in predicts]
    index.extend(fnames)
    results.extend(predicts)
    hol = hol +64

df = pd.DataFrame(columns=['fname', 'label'])
df['fname'] = index
df['label'] = results
df.to_csv(os.path.join(out_path, 'sub18.csv'), index=False)

file = open("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/sub18.csv",'r')
str = file.read()
valami = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
for s in valami:
    str = str.replace(','+s, ",unknown")
str = str.replace("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/test/audio/" ,"")
file.close()
file = open("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/sub18_corr.csv",'w')
file.write(str)
file.close()