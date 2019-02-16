#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:00:09 2017

@author: gabor
"""

import numpy as np
from keras.models import load_model
import os, librosa
from sklearn.externals import joblib
#%%
def reshape(array, size=[20,101]):
    s=array.shape
    if(s != size):
        ret = np.zeros(size)
        ret[0:s[0],0:s[1]]=array
    return ret
#%%
target_names = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
#%%
scaler = joblib.load("/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/scaler_40dd_wolabel.pkl")
path="/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/"
model = load_model(path + "scripts/model_40dd_wolabel_2")
X_test=np.zeros([1,120,101,1])
files = os.listdir(path + "test/audio")# list dirs
tofile="fname,label"
i = 0
for file in files: #for every file
    i += 1
    filename = path + "test/audio/" + file
    y, sr = librosa.load(filename) #read in the file
    inputs = reshape(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, hop_length=int(sr/100), n_fft=int(sr/40),fmin=300, fmax=4000),size=[40,101])
    mfcc_delta = librosa.feature.delta(inputs)
    mfcc_delta2 = librosa.feature.delta(inputs, order=2)
    inputs=np.concatenate((inputs,mfcc_delta, mfcc_delta2),axis=0)
    shape = inputs.shape
    tmp = np.reshape(inputs, -1, order="F")
    tmp = scaler.transform(tmp.reshape(1, -1))
    inputs = np.reshape(tmp, shape, order="F")
    X_test[0,:,:,:]=np.atleast_3d(inputs)
    pred = model.predict(X_test)
    lab = np.argmax(pred)
    tofile += "\n" + file + ","
    tofile += target_names[lab]
    if i%2000 == 0:
        print(i)
# end of foreach files
#%%
file = open(path + "7_attempt.csv", "w")
file.write(tofile)
file.close()