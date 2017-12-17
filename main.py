# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:18:14 2017

@author: Michał
"""
import gc
import numpy as np
import network as net
import DataExtractor as de
from skimage import io

dim=(8,8,1)

if dim[2]==1:
    g=True
else:
    g=False

model=net.NeuralNetwork()
extractor=de.DataExtractor()

trainTemp=extractor.extractData("DATA/mask/01_h.jpg","DATA/mask/01_h.tif",shape=dim,grey=g)

train=(np.reshape(trainTemp[0],(trainTemp[0].shape[0],dim[2],dim[0],dim[1])),
       trainTemp[1])

del trainTemp
model.create(size=dim)
model.learn(train,epoch=10)


trainTemp=extractor.extractData("DATA/mask/02_h.jpg","DATA/mask/02_h.tif",shape=dim,grey=g)

train=(np.reshape(trainTemp[0],(trainTemp[0].shape[0],dim[2],dim[0],dim[1])),
       trainTemp[1])



model.learn(train,epoch=10)


trainTemp=extractor.extractData("DATA/mask/03_h.jpg","DATA/mask/03_h.tif",shape=dim,grey=g)

train=(np.reshape(trainTemp[0],(trainTemp[0].shape[0],dim[2],dim[0],dim[1])),
       trainTemp[1])



model.learn(train,epoch=10)



#print("")

#io.imshow(np.reshape(train[1][2],(400,400)))


del train
del model

gc.collect()

#print(model.evaluate(train))