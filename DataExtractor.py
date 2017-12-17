# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:59:48 2017

@author: Michał
"""
import numpy as np
from skimage import io


class DataExtractor:
    
    def extractSingle(self,img,shape=(150,150),coor=(0,0)):
        
        return img[coor[0]:coor[0]+shape[0],coor[1]:coor[1]+shape[1]]
    
    def extractData(self,photo,mask,shape=(150,150),grey=False):
        
        img=io.imread(photo,as_grey=grey)
        
        imgMask=io.imread(mask)
        
        dataPhoto=[]
        dataMask=[]
        rowsQty=int(len(img)/shape[0])
        colsQty=int(len(imgMask)/shape[1])
      
        for i in range(0,rowsQty):
            for j in range(0,colsQty):
                pos=(i*shape[0],j*shape[1])
                dataPhoto.append(self.extractSingle(img,shape=shape,coor=pos))
                dataMask.append(np.ndarray.flatten(self.extractSingle(imgMask,shape=shape,coor=pos)/255))
               
        del img
        del imgMask
        
        return (np.asarray(dataPhoto),np.asarray(dataMask))   
                

#ext=DataExtractor()


#dd=ext.extractData("DATA/mask/01_h.jpg","DATA/mask/01_h.tif",shape=(50,50))


#print(dd[0].shape)