#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 15:06:43 2017

@author: gabor
"""

import os
import numpy as np
import librosa
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy import signal
from glob import glob
import re
import pandas as pd
import gc
from scipy.io import wavfile
from sklearn import preprocessing

from keras import optimizers, losses, activations, models
from keras.layers import Convolution2D, Dense, Input, Flatten, Dropout, MaxPooling2D, BatchNormalization
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

def chop_audio(samples, L=16000, num=400):
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
"""new_sample_rate = 16000
sample_rate, samples = wavfile.read("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/train/audio/bed/"+fnames[0])
samples = pad_audio(samples)
resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
_, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
cepstrum = real_cepstrum(specgram)"""
#%%
new_sample_rate = 16000
y_train = []
x_train = []
shapes = []
i=0
for label, fname in zip(labels, fnames):
    sample_rate, samples = wavfile.read(os.path.join(train_data_path, label, fname))
    samples = pad_audio(samples)
    if i%1000==0:
        print(i)
    i=i+1
    if len(samples) > 16000:
        n_samples = chop_audio(samples)
    else: n_samples = [samples]
    for samples in n_samples:
        resampled = signal.resample(samples, int(new_sample_rate / sample_rate * samples.shape[0]))
        #_, _, specgram = log_specgram(resampled, sample_rate=new_sample_rate)
        inputs = librosa.feature.mfcc(y=resampled, sr=new_sample_rate, n_mfcc=40, hop_length=int(new_sample_rate/100), n_fft=int(new_sample_rate/40),fmin=300, fmax=8000)
        mfcc_delta = librosa.feature.delta(inputs)
        mfcc_delta2 = librosa.feature.delta(inputs, order=2)
        inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)
        y_train.append(label)
        x_train.append(inputs)
shape = x_train[0].shape
#%%
"""x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2017)
for i in range(len(x_train)):
    x_train[i]=np.reshape(x_train[i], -1, order="F")
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
#%%
for i in range(len(x_valid)):
    x_valid[i]=np.reshape(x_valid[i], -1, order="F")
x_valid = scaler.transform(x_valid)
from sklearn.externals import joblib
joblib.dump(scaler, "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/models/scaler15.pkl")
#%%
x_train = list(x_train)
#%%
for i in range(len(x_train)):
    x_train[i]=np.reshape(x_train[i], shape, order="F")
#%%
x_valid = list(x_valid)
for i in range(len(x_valid)):
    x_valid[i]=np.reshape(x_valid[i], shape, order="F")"""
#%%
x_train = np.array(x_train)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
y_train = label_transform(y_train)
label_index = y_train.columns.values
y_train = y_train.values
y_train = np.array(y_train)
del labels, fnames
gc.collect()      
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=2017)  
#%%
print(x_train[4,:,:,0])
#%%
from keras.callbacks import ModelCheckpoint, EarlyStopping
model_path = "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/models"
patience=15
early_stopping=EarlyStopping(patience=patience, verbose=1)
checkpointer=ModelCheckpoint(os.path.join(model_path, 'cnn_mfcc2.model'), save_best_only=True, verbose=1)
input_shape = (120, 101, 1)
nclass = 31
inp = Input(shape=input_shape)
img_1 = BatchNormalization()(inp)
img_1 = Convolution2D(22, kernel_size=2, activation=activations.relu)(img_1)
img_1 = Convolution2D(22, kernel_size=2, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(0.25)(img_1)
img_1 = Convolution2D(36, kernel_size=3, activation=activations.relu)(img_1)
img_1 = Convolution2D(36, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(0.25)(img_1)
img_1 = Convolution2D(75, kernel_size=3, activation=activations.relu)(img_1)
img_1 = Convolution2D(75, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(0.25)(img_1)
img_1 = Convolution2D(140, kernel_size=3, activation=activations.relu)(img_1)
img_1 = Convolution2D(140, kernel_size=3, activation=activations.relu)(img_1)
img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
img_1 = Dropout(0.25)(img_1)
img_1 = Flatten()(img_1)

dense_1 = BatchNormalization()(Dense(200, activation=activations.relu)(img_1))
dense_1 = BatchNormalization()(Dense(128, activation=activations.relu)(dense_1))
dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

model = models.Model(inputs=inp, outputs=dense_1)
opt = optimizers.Adam()

model.compile(optimizer=opt, loss=losses.binary_crossentropy)                                                                                                                                                                                                                                                                                               
model.summary()
#%%
model.fit(x_train, y_train, batch_size=64, validation_data=(x_valid, y_valid), epochs=1000, callbacks=[checkpointer, early_stopping],shuffle=True, verbose=1)
#%%
"""from keras.models import load_model
model_path = "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/models"
model = load_model(os.path.join(model_path, 'cnn5.model'))
model.summary()"""
#%%
from keras.models import load_model
model = load_model(os.path.join(model_path, 'cnn_mfcc2.model'))
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
        resampled = signal.resample(samples, int(new_sample_rate / rate * samples.shape[0]))
        inputs = librosa.feature.mfcc(y=resampled, sr=new_sample_rate, n_mfcc=40, hop_length=int(new_sample_rate/100), n_fft=int(new_sample_rate/40),fmin=300, fmax=8000)
        mfcc_delta = librosa.feature.delta(inputs)
        mfcc_delta2 = librosa.feature.delta(inputs, order=2)
        inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)
        imgs.append(inputs)
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
df.to_csv(os.path.join(out_path, 'sub_mfcc2.csv'), index=False)

file = open("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/sub_mfcc2.csv",'r')
str = file.read()
valami = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow"]
for s in valami:
    str = str.replace(','+s, ",unknown")
str = str.replace("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/test/audio/" ,"")
file.close()
file = open("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/sub_mfcc2_corr.csv",'w')
file.write(str)
file.close()