# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:36:05 2017

@author: Michał
"""
import numpy as np
from keras import backend as K
K.set_image_dim_ordering('th')
import os 
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils


class NeuralNetwork():
    
    def __init__(self):
        self.net=Sequential()
    
    def create(self,size=(28,28,1)):
        
        self.net.add(Conv2D(32, 2, 2, activation='relu', input_shape=(size[2],size[0],size[1])))
       
        self.net.add(MaxPooling2D(pool_size=(2,2)))
        
        self.net.add(Conv2D(16, 2, 2, activation='relu', input_shape=(size[2],size[0],size[1])))
       
        self.net.add(MaxPooling2D(pool_size=(2,2)))
        
        self.net.add(Dropout(0.5))
 
        self.net.add(Flatten())

        self.net.add(Dense(size[0]*size[1], activation='relu'))
        self.net.compile(loss='mean_squared_error',optimizer='adam',metrics=['accuracy'])
        
    def learn(self,data,epoch=1):
        self.net.fit(data[0], data[1],batch_size=32, nb_epoch=epoch, verbose=1)
    

    def load(self,file):
        
        if os.path.isfile(file):
            self.net=load_model(file)
            
            print("File loaded...")
        else:
            print("File do not exist...")
    
    def evaluate(self,data):
        return self.net.evaluate(data[0], data[1], verbose=1)
        
    def predict(self,inputData,outputShape):
        return np.reshape(self.net.predict(inputData,batch_size=1,verbose=1),outputShape)