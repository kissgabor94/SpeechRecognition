#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 21:43:06 2017

@author: gabor
"""
import sys
import numpy as np
import glob
from scipy.io import wavfile
from python_speech_features import mfcc
import random
from sklearn import preprocessing

random.seed(123)
dataset = []

trainpath=str(sys.argv[1])
if trainpath[-1]=="/": # if the path end with /
    trainpath +="???_??_?.wav"
else: # if the path do not end with /
    trainpath +="/???_??_?.wav"


testpath=str(sys.argv[2])
if testpath[-1]=="/": # if the path end with /
    testpath +="???_??_?.wav"
else: # if the path do not end with /
    testpath +="/???_??_?.wav"
    

validpath=str(sys.argv[3])
if validpath[-1]=="/": # if the path end with /
    validpath +="???_??_?.wav"
else: # if the path do not end with /
    validpath +="/???_??_?.wav"

#files = glob.glob("/home/gabor/Audio/Train/???_??_?.wav") #get the list of the file names with proper format (the sound files)
files = glob.glob(trainpath) #get the list of the file names with proper format (the sound files)
for name in files: #for every file
    freq, data = wavfile.read(name) #read in the wav file
    #calculating MFCC with 2048 sized FFT, 20 cepstrums, 4kHz maximum and 300Hz minimum mel filter frequency and hanning window
    inputs = mfcc(data[:,0], samplerate=freq, nfft=2048, numcep=20, highfreq=4000, lowfreq=300, winfunc=np.hanning)
    outputs = np.zeros([1,15])
    outputs[0][int(name[-8]+name[-7])-1]=1 # set the adequate output to 1, the others stay 0
    temp = []
    temp.append(inputs)
    temp.append(outputs)
    dataset.append(temp) #the dataset is a list of lists 
    
#files = glob.glob("/home/gabor/Audio/Valid/???_??_?.wav") #get the list of the file names with proper format (the sound files)
files = glob.glob(validpath) #get the list of the file names with proper format (the sound files)
for name in files: #for every file
    freq, data = wavfile.read(name) #read in the wav file
    #calculating MFCC with 2048 sized FFT, 20 cepstrums, 4kHz maximum and 300Hz minimum mel filter frequency and hanning window
    inputs = mfcc(data[:,0], samplerate=freq, nfft=2048, numcep=20, highfreq=4000, lowfreq=300, winfunc=np.hanning)
    outputs = np.zeros([1,15])
    outputs[0][int(name[-8]+name[-7])-1]=1 # set the adequate output to 1, the others stay 0
    temp = []
    temp.append(inputs)
    temp.append(outputs)
    dataset.append(temp) #the dataset is a list of lists 
    
#files = glob.glob("/home/gabor/Audio/Test/???_??_?.wav") #get the list of the file names with proper format (the sound files)
files = glob.glob(testpath) #get the list of the file names with proper format (the sound files)
for name in files: #for every file
    freq, data = wavfile.read(name) #read in the wav file
    #calculating MFCC with 2048 sized FFT, 20 cepstrums, 4kHz maximum and 300Hz minimum mel filter frequency and hanning window
    inputs = mfcc(data[:,0], samplerate=freq, nfft=2048, numcep=20, highfreq=4000, lowfreq=300, winfunc=np.hanning)
    outputs = np.zeros([1,15])
    outputs[0][int(name[-8]+name[-7])-1]=1 # set the adequate output to 1, the others stay 0
    temp = []
    temp.append(inputs)
    temp.append(outputs)
    dataset.append(temp) #the dataset is a list of lists 
    
random.shuffle(dataset) #shuffle the dataset
train = []
valid = []
test = []
train = dataset[:int(len(dataset)*0.7)] #train split
valid = dataset[int(len(dataset)*0.7):int(len(dataset)*0.9)] #valid split
test = dataset[int(len(dataset)*0.9):] #test split 

temp_train = []
temp_valid = []
temp_test = []
# Here I reshape the inputs to 1D arrays (this is necessary for the standardization)
for item in train:
    temp_train.append(np.reshape(item[0], -1, order="F")) #do not change the order
temp_valid = []
for item in valid:
    temp_valid.append(np.reshape(item[0],-1, order="F")) #do not change the order
temp_test = []
for item in test:
    temp_test.append(np.reshape(item[0], -1, order="F")) #do not change the order

scaler = preprocessing.StandardScaler().fit(temp_train) #calculate the scaler from the train data
#scale the train, validation and test datasets
temp_train = scaler.transform(temp_train)
temp_valid = scaler.transform(temp_valid)
temp_test = scaler.transform(temp_test)

#Here I reshape the scaled data to the original shape
shape = np.shape(train[0][0]) #original shape
for i in range(len(train)):
    train[i][0] = np.reshape(temp_train[i,:], shape, order="F")
for i in range(len(valid)):
    valid[i][0] = np.reshape(temp_valid[i,:], shape, order="F")
for i in range(len(test)):
    test[i][0] = np.reshape(temp_test[i,:], shape, order="F")

#Here I save the collected data into files. Lower I show an example how to read them back in proper format
np.save(sys.argv[4], train)
np.save(sys.argv[6], valid)
np.save(sys.argv[5], test)

#Here I show how to read the saved files back
"""
data=[]
from_file=np.load("file.npy") #this returns an object variable
for i in range(np.shape(from_file)[0]): #for every row in the object
    tmp=[]
    for j in range(np.shape(from_file)[1]): #for every column in the object. It will have only 2: inputs and outputs, but better be safe
        tmp.append(from_file[i][j]) #collect the inputs and outputs in a list for every row
    data.append(tmp) #append the list with the new sample
"""