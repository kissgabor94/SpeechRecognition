#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 09:34:44 2017

@author: gabor
"""

import numpy as np
from keras.models import Sequential
from keras import regularizers
from keras.layers.core import  Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix
import gc
#%%
gc.collect()
#%%
path="/media/gabor/ALL/MachineLearning/kaggle/TensorFlow_Speech_Recognition/scripts/"
train = np.load(path + "train_specgram_alllabels_wolabel.npz")
X_train = train['inputs']
Y_train = train['outputs']
train = None

shape = X_train.shape[1:3]
valid = np.load(path + "valid_specgram_alllabels_wolabel.npz")
X_valid = valid['inputs']
Y_valid = valid['outputs']
valid = None
test = np.load(path + "test_specgram_alllabels_wolabel.npz")
X_test = test['inputs']
Y_test = test['outputs']
test = None

#%%
patience=20
early_stopping=EarlyStopping(patience=patience, verbose=1)

checkpointer=ModelCheckpoint(path+"model_specgram_alllabels_5", save_best_only=True, verbose=1)
#%%
reg=0.0001
model= Sequential()
model.add(Conv2D(11, (3, 3), padding='same',  input_shape=(shape[0], shape[1], 1)))
model.add(Conv2D(22, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(22, (3, 3), padding='same'))
model.add(Conv2D(33, (3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(33, (3, 3), padding='same'))
model.add(Conv2D(44, (3, 3), padding='same'))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(31, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
#%%

model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_valid, Y_valid),callbacks=[checkpointer, early_stopping], shuffle=True)
#%%
model = load_model(path+"model_specgram_alllabels_5")
score = model.evaluate(X_test, Y_test, verbose=0) #calculate error for the test dataset
print("The score of the test is ", (score))
Y_pred = model.predict(X_test) #predicitions
y_pred = np.argmax(Y_pred, axis=1) #indicies of the "tips"

p=model.predict_proba(X_test) # to predict probability
target_names = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow", "silence"]
print("\n",classification_report(np.argmax(Y_test,axis=1), y_pred,target_names=target_names)) #classification report
conf_mat=confusion_matrix(np.argmax(Y_test,axis=1), y_pred) #confusion matrix
summ=0
for i in range(conf_mat.shape[0]): #calculte the number of accurate predictions
    summ = summ + conf_mat[i,i] 
print("Accuracy: ", (summ/conf_mat.sum()*100), "%") #and divide it by the number of all predicitons (accuracy in %)

#%%
model.summary() #print model summary


