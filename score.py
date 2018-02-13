# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:57:14 2018

@author: Michał
"""

class Score():
    
    def __init__(self,f_true,f_pred,f_class):
        self.ground_truth=f_true
        self.prediction=f_pred
        self.f_class=f_class
        self.tp=0
        self.tn=0
        self.fp=0
        self.fn=0
        self.count=0
        
    
    def recall(self):
        
        if self.tp+self.fn==0:
            return 0
        
        return self.tp/(self.tp+self.fn)
    
    
    def precision(self):
        
        if self.tp+self.fp==0:
            return 0
        
        return self.tp/(self.tp+self.fp)
    
    
    def accuracy(self):
        
        if self.tp+self.tn+self.fp+self.fn==0:
            return 0
        
        return (self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)  
    
    def f1_score(self):
        
        recall=self.recall()
        precision=self.precision()
        
        if recall+precision==0:
            return 0
        
        return 2*(recall*precision)/(recall+precision) 
    
    def calculate(self):
        
        for pair in zip(self.ground_truth,self.prediction):
            
            if pair[1]!=0 and pair[1]!=255:
                
                print("babol"+str(pair[1]))
            
            if pair[1]==self.f_class:
                self.count=self.count+1
                
                if pair[1]==pair[0]:
                    self.tp=self.tp+1
                else:
                    self.fp=self.fp+1
            else:
                    
                if pair[0]==self.f_class:
                    self.fn=self.fn+1               
                else:
                    self.tn=self.tn+1
                    
        
        print("score calculated")
        



