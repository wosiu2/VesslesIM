# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:18:14 2017

@author: Michał
"""
import numpy as np
import network as net
import DataExtractor as de

dim=(100,100)

model=net.NeuralNetwork()
extractor=de.DataExtractor()

trainTemp=extractor.extractData("DATA/mask/01_h.jpg","DATA/mask/01_h.tif",shape=dim)

train=(np.reshape(trainTemp[0],(trainTemp[0].shape[0],3,dim[0],dim[1])),
       trainTemp[1])

model.create(train,size=dim,epoch=1)

#print(model.evaluate(train))