#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:44:17 2017

@author: gabor
"""

import random, os
import numpy as np
from sklearn import preprocessing
import librosa
import gc
from sklearn.externals import joblib
#%%
def reshape(array, size=[20,101]):
    s=array.shape
    if(s != size):
        ret = np.zeros(size)
        ret[0:s[0],0:s[1]]=array
    return ret
#%%
labels = {"yes": 0, "no": 1, "up": 2, "down": 3, "left": 4, "right": 5, "on": 6, "off": 7, "stop": 8, "go": 9, "zero": 10, "one": 10, "two": 10, "three": 10, "four": 10, "five": 10, "six": 10, "seven": 10, "eight": 10, "nine": 10, "bed": 10, "bird": 10, "cat": 10, "dog": 10, "happy": 10, "house": 10, "marvin": 10, "sheila": 10, "tree": 10, "wow": 10, "silence": 11}
#%%
test = []
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
        i=0
        for file in files: # for every file in the folder
            substr = file[0:8]
            if substr > "1fd85ee4" and substr <= "3ab9ba07":
                if i%3==0:
                    filename = root + "/" + folder + "/" + file
                    y, sr = librosa.load(filename) #read in the file
                    inputs = reshape(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=4000),size=[40,101])
                    mfcc_delta = librosa.feature.delta(inputs)
                    mfcc_delta2 = librosa.feature.delta(inputs, order=2)
                    inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)
                    outputs = np.zeros(12)
                    outputs[index]=1
                    temp = []
                    temp.append(inputs)
                    temp.append(outputs)
                    test.append(temp)
                i = i+1
            """if substr <= "1fd85ee4":
                valid.append(temp) 
            elif substr <= "3ab9ba07":
                test.append(temp)
            else:
                train.append(temp)"""
        # foreach files
    #if not background noise
#foreach dirs
dirs = None
#%% create silence clips
def onesecs(clip):
    l=len(clip)
    done = True
    i=0
    ret=[]
    while(done):
        if ((i+22050)<l):       
            tmp = clip[i:i+22050]
            ret.append(tmp)
            i+=11025
        else:
            done = False
    return ret

base = "/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/train/audio/_background_noise_/"
something = [base + "doing_the_dishes.wav", base + "dude_miaowing.wav", base + "exercise_bike.wav", base + "pink_noise.wav", base + "running_tap.wav", base + "white_noise.wav"]
for i in range(6):
    if i==5:
        y, sr = librosa.load(something[i])
        clips = onesecs(y)
        for f in clips:
            inputs = reshape(librosa.feature.mfcc(y=f, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=4000),size=[40,101])
            mfcc_delta = librosa.feature.delta(inputs)
            mfcc_delta2 = librosa.feature.delta(inputs, order=2)
            inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)
            outputs = np.zeros(12)
            outputs[11]=1
            temp = []
            temp.append(inputs)
            temp.append(outputs)
            test.append(temp)
#%% standardization 
random.shuffle(test) #shuffle the dataset
shape = np.shape(test[0][0]) #original shape
X_test = np.zeros((len(test), shape[0], shape[1], 1)) #pre-allocate np matrices for the data
Y_test = np.zeros((len(test), 12)) #and the labels

temp_test = []
# Here I reshape the inputs to 1D arrays (this is necessary for the standardization)
for item in test:
    temp_test.append(np.reshape(item[0], -1, order="F")) #do not change the order

scaler = joblib.load("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/scaler_40dd_wolabel_trunc.pkl")
#scale the train, validation and test datasets
temp_test = scaler.transform(temp_test)

#Here I reshape the scaled data to the original shape
for i in range(len(test)):
    X_test[i,:,:,:]=np.atleast_3d(np.reshape(temp_test[i,:], shape, order="F"))
    Y_test[i,:]=test[i][1]

temp_test = None
test = None
gc.collect()
path="/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/"
np.savez_compressed(path+"test_40dd_wolabel_trunc.npz", inputs = X_test, outputs=Y_test)
