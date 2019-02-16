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
labels = {"yes": 0, "no": 1, "up": 2, "down": 3, "left": 4, "right": 5, "on": 6, "off": 7, "stop": 8, "go": 9, "silence": 10, "unknown": 11 }
#%%
valid = []
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
            if substr <= "1fd85ee4":
                filename = root + "/" + folder + "/" + file
                y, sr = librosa.load(filename) #read in the file
                inputs = reshape(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=4000),size=[20,101])
                mfcc_delta = librosa.feature.delta(inputs)
                mfcc_delta2 = librosa.feature.delta(inputs, order=2)
                inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)
                outputs = np.zeros(12)
                outputs[index]=1
                temp = []
                temp.append(inputs)
                temp.append(outputs)
                valid.append(temp)
            """if substr <= "1fd85ee4":
                valid.append(temp) 
            elif substr <= "3ab9ba07":
                test.append(temp)
            else:
                train.append(temp)"""
        # foreach files
    #if not background noise
#foreach dirs
del dirs
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
    if i==4:
        y, sr = librosa.load(something[i])
        clips = onesecs(y)
        for f in clips:
            inputs = reshape(librosa.feature.mfcc(y=f, sr=sr, n_mfcc=20, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=4000),size=[20,101])
            mfcc_delta = librosa.feature.delta(inputs)
            mfcc_delta2 = librosa.feature.delta(inputs, order=2)
            inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)
            outputs = np.zeros(12)
            outputs[10]=1
            temp = []
            temp.append(inputs)
            temp.append(outputs)
            valid.append(temp)
#%% standardization 
random.shuffle(valid) #shuffle the dataset

temp_valid = []
# Here I reshape the inputs to 1D arrays (this is necessary for the standardization)
for item in valid:
    temp_valid.append(np.reshape(item[0], -1, order="F")) #do not change the order

scaler = joblib.load("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/scaler_40dd.pkl")
#scale the train, validation and test datasets
temp_valid = scaler.transform(temp_valid)

#Here I reshape the scaled data to the original shape
shape = np.shape(valid[0][0]) #original shape
for i in range(len(valid)):
    valid[i][0] = np.reshape(temp_valid[i,:], shape, order="F")

path="/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/"
np.save(path+"valid_40dd.npy", valid)