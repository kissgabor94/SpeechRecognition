#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:34:44 2017

@author: gabor
"""

import sys
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
#%%
data=[]
#from_file=np.load("/home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/traindata_sep.npy") #this returns an object variable
from_file=np.load(str(sys.argv[1])) #this returns an object variable
for i in range(np.shape(from_file)[0]): #for every row in the object
    tmp=[]
    for j in range(np.shape(from_file)[1]): #for every column in the object. It will have only 2: inputs and outputs, but better be safe
        tmp.append(from_file[i][j]) #collect the inputs and outputs in a list for every row
    data.append(tmp) #append the list with the new sample

shape=data[0][0].shape #get the shape of the input

X_train = np.zeros((len(data), shape[0], shape[1])) #pre-allocate np matrices for the data
Y_train = np.zeros((len(data), 15)) #and the labels

for i in range(len(data)):
    X_train[i,:,:]=(data[i][0])
    Y_train[i,:]=np.reshape(data[i][1], 15)


data=[]
#from_file=np.load("/home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/testdata_sep.npy") #this returns an object variable
from_file=np.load(str(sys.argv[2])) #this returns an object variable
for i in range(np.shape(from_file)[0]): #for every row in the object
    tmp=[]
    for j in range(np.shape(from_file)[1]): #for every column in the object. It will have only 2: inputs and outputs, but better be safe
        tmp.append(from_file[i][j]) #collect the inputs and outputs in a list for every row
    data.append(tmp) #append the list with the new sample

X_test = np.zeros((len(data), shape[0], shape[1])) #pre-allocate np matrices for the data
Y_test = np.zeros((len(data), 15)) #and the labels

for i in range(len(data)):
    X_test[i,:,:]=(data[i][0])
    Y_test[i,:]=np.reshape(data[i][1], 15)
    
data=[]
#from_file=np.load("/home/gabor/DeepLearningHomework/vitmav45-2017-DeepImpact/validdata_sep.npy") #this returns an object variable
from_file=np.load(str(sys.argv[3])) #this returns an object variable
for i in range(np.shape(from_file)[0]): #for every row in the object
    tmp=[]
    for j in range(np.shape(from_file)[1]): #for every column in the object. It will have only 2: inputs and outputs, but better be safe
        tmp.append(from_file[i][j]) #collect the inputs and outputs in a list for every row
    data.append(tmp) #append the list with the new sample

X_valid = np.zeros((len(data), shape[0], shape[1])) #pre-allocate np matrices for the data
Y_valid = np.zeros((len(data), 15)) #and the labels

for i in range(len(data)):
    X_valid[i,:,:]=(data[i][0])
    Y_valid[i,:]=np.reshape(data[i][1], 15)
#%%
patience=30
early_stopping=EarlyStopping(patience=patience, verbose=1)

checkpointer=ModelCheckpoint(filepath=sys.argv[4], save_best_only=True, verbose=1)
#%%
model= Sequential()
model.add(Conv1D(kernel_size=5, padding='same', filters=25 ,input_shape=(shape[0], shape[1]) ))
model.add(Dropout(0.4))
model.add(Conv1D(kernel_size=5, padding='same', filters=20))
model.add(Dropout(0.4))
model.add(MaxPooling1D(pool_size=4))
model.add(Conv1D(kernel_size=5, padding='same', filters=15))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(LSTM(units=750, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=500, return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(units=350, return_sequences=False))
model.add(Dropout(0.4))
#model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(15, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
#%%

model.fit(X_train, Y_train, batch_size=32, epochs=1000, validation_data=(X_valid, Y_valid),callbacks=[checkpointer, early_stopping])
#%%
model = load_model(sys.argv[4])
score = model.evaluate(X_test, Y_test, verbose=0) #calculate error for the test dataset
print("The score of the test is ", (score))
Y_pred = model.predict(X_test) #predicitions
y_pred = np.argmax(Y_pred, axis=1) #indicies of the "tips"

p=model.predict_proba(X_test) # to predict probability
target_names = ['menj', 'gyere', 'tölts', 'Ethon', 'vezess', 'keress', 'szia', 'vissza', 'nyisd', 'viszlát', 'Bea', 'Ádám', 'Márta', 'Antal', 'Bence']
print("\n",classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names)) #classification report
conf_mat=confusion_matrix(np.argmax(Y_test,axis=1), y_pred) #confusion matrix
summ=0
for i in range(conf_mat.shape[0]): #calculte the number of accurate predictions
    summ = summ + conf_mat[i,i] 
print("Accuracy: ", (summ/conf_mat.sum()*100), "%") #and divide it by the number of all predicitons (accuracy in %)

#%%
model.summary() #print model summary
#%%



