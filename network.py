# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:36:05 2017

@author: Michał
"""
import numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')
import os 
from keras.models import Model
from keras.layers import merge
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D,Input
from keras.optimizers import Adam



smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

class NeuralNetwork():
    
    def __init__(self):

        self.net=Model(inputs=[], outputs=[])

        
        
    def save(self,file):
        self.net.save(file)
        
    def save_weights(self,file):
        self.net.save_weights(file)
        
        
    def create(self,size=(28,28,1)):
        inputs=Input((size[2],size[0],size[1]))
        #self.net.add(Conv2D(64, (4,4), activation='relu',padding='same', input_shape=(size[2],size[0],size[1])))
        #self.net.add(MaxPooling2D(pool_size=(2,2)))
        #self.net.add(Conv2D(32, (3,3),padding='same', activation='relu'))
        #self.net.add(MaxPooling2D(pool_size=(2,2)))
        #self.net.add(Conv2D(16, (3,3),padding='same', activation='relu'))
        #self.net.add(MaxPooling2D(pool_size=(2,2)))
        
        #self.net.add(Conv2D(16, (3,3),padding='same', activation='relu'))
        #self.net.add(UpSampling2D(size=(2,2)))
        
        #self.net.add(Conv2D(32, (3,3),padding='same', activation='relu'))
        #self.net.add(UpSampling2D(size=(2,2)))        
        
        #self.net.add(Conv2D(64,(4,4),padding='same', activation='relu'))
        #self.net.add(UpSampling2D(size=(2,2)))
        #self.net.add(Conv2D(32, 2, 2, activation='relu', input_shape=(size[2],size[0],size[1])))
       
        #self.net.add(MaxPooling2D(pool_size=(2,2)))
        
        #self.net.add(Conv2D(16, 2, 2, activation='relu', input_shape=(size[2],size[0],size[1])))
       
       # self.net.add(MaxPooling2D(pool_size=(2,2)))
        
        #self.net.add(Dropout(0.5))
 
       # self.net.add(Flatten())

       # self.net.add(Dense(size[0]*size[1], activation='relu'))
        
        #self.net.add(Conv2D(1,(1,1),padding='same',activation='softmax'))
        
        
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
        
        #up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
        
       # up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
        up7=merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)

        #up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
        
        #up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        self.net = Model(inputs=[inputs], outputs=[conv10])
        
        
        
        self.net.compile(loss=dice_coef_loss,optimizer=Adam(lr=1e-5),metrics=[dice_coef])
        
    def learn(self,data,epoch=1):
        self.net.fit(data[0], data[1],batch_size=2, nb_epoch=epoch, verbose=1)
    

    def load(self,size,file):
        
        if os.path.isfile(file):
            
            #self.net=load_model(file)
            self.create(size)
            self.net.load_weights(file)
            
            print("File loaded...")
        else:
            print("File do not exist...")
    
    def evaluate(self,data):
        return self.net.evaluate(data[0], data[1], verbose=1)
        
    def predict(self,inputData,outputShape):
        return np.reshape(self.net.predict(inputData,batch_size=1,verbose=0),outputShape)
    
    def predictSet(self,data,shape,frame):
        
        result = np.zeros((frame[0]*shape[0],frame[1]*shape[1]))
        
        for i in range(shape[0]):
   
            for j in range(shape[1]):
       
                arr=data[0][i*shape[1]+j][0]
       
                arr2=np.resize(arr,(1,1,frame[0],frame[1]))
                im=np.reshape(self.predict(arr2,frame),(frame[0],frame[1]))
                
                result[i*frame[0]:(i+1)*frame[0],j*frame[1]:(j+1)*frame[1]]=im
                os.system('cls')
                print(str(i*shape[1]+j+1)+"/"+str(shape[0]*shape[1]))
        
        return result
    
    
    