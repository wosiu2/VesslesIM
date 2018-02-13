# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 19:59:48 2017

@author: Michał
"""
import numpy as np
from skimage import io
import math

class DataExtractor:
    def reframe(self,data,frame):
    
        rows=math.ceil(len(data)/frame[0])
        cols=math.ceil(len(data[0])/frame[1])
    
        tmp=np.zeros((rows*frame[0],cols*frame[1]))
    
        tmp[0:data.shape[0],0:data.shape[1]]=data
    
        return (tmp,(rows,cols))

    def extractSingle(self,img,shape=(150,150),coor=(0,0)):
        
        return img[coor[0]:coor[0]+shape[0],coor[1]:coor[1]+shape[1]]
    
    def extractData(self,photo,mask,shape=(150,150),grey=False,channel=0):
        tmpImg=io.imread(photo,as_grey=grey)
        
        if grey==True:
            img=tmpImg
        else:
            img=tmpImg[:,:,channel]    
        
        if mask==None:
            imgMask=np.zeros(img.shape)
        else:
            imgMask=io.imread(mask)
        #print(img.shape)
        #io.imshow(img)
        
        
        dataPhoto=[]
        dataMask=[]
        
        imgMask_t=self.reframe(imgMask,shape)
        img_t=self.reframe(img,shape)
        
        rowsQty=img_t[1][0]
        colsQty=img_t[1][1]
      
        for i in range(0,rowsQty):
            for j in range(0,colsQty):
                pos=(i*shape[0],j*shape[1])
                dataPhoto.append(self.extractSingle(img_t[0],shape=shape,coor=pos))
                dataMask.append(self.extractSingle(imgMask_t[0],shape=shape,coor=pos)/255)
                #np.ndarray.flatten()
        del img
        del imgMask
        del img_t
        del imgMask_t
        
        trainTemp=(np.asarray(dataPhoto),np.asarray(dataMask))
        
        train=(np.reshape(trainTemp[0],(trainTemp[0].shape[0],shape[2],shape[0],shape[1])),
               np.reshape(trainTemp[1],(trainTemp[1].shape[0],shape[2],shape[0],shape[1])))
        
        return train
    
    def shape2(self,file,shape):
        
        img=io.imread(file,as_grey=True)
               
        
        return (math.ceil(len(img)/shape[0]),math.ceil(len(img[0])/shape[1]))
    
    def cutOff(self, arr,ratio):
        tmp=np.copy(arr)
        
        tmp[tmp<ratio]=0    
        tmp[tmp>=ratio]=1
        
        return tmp
        
    def expand(self,img,div):
        
        
        return 1
                 

#ext=DataExtractor()


#dd=ext.extractData("DATA/mask/01_h.jpg","DATA/mask/01_h.tif",shape=(50,50))


#print(dd[0].shape)