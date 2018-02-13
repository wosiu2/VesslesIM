import numpy as np
from skimage import io
import os
from sklearn.metrics import accuracy_score, jaccard_similarity_score,classification_report,precision_score
from score import Score

def dice_coef(y_true_f, y_pred_f,smooth=1.0):

    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f*y_true_f) + np.sum(y_pred_f*y_pred_f) + smooth)

def unifyShape(input_array,ref_array):
    
    rows=np.minimum(input_array.shape[0],ref_array.shape[0])
    cols=np.minimum(input_array.shape[1],ref_array.shape[1])
    
    temp_array=input_array[0:rows,0:cols]
    
    return temp_array
def normalize(matrix):
    temp=np.zeros(matrix.shape)
    print(1)
    for i in range(matrix.shape[0]):
            temp[i]=int(matrix[i]/255)
    return temp
def precision(f_true,f_pred,val=None):
    tp=0
    fp=0
    
    for i in zip(f_true,f_pred):
        
        if i[0]==i[1]:
            if val==None:
                tp=tp+1
            if val==i[0]:
                tp=tp+1            
        else:
            if val==None:
                fp=fp+1
            if val==i[1]:
                fp=fp+1
            
    print(tp)
    print(fp)
    if fp+tp==0:
        return 0
    
    return tp/(fp+tp)
    

def openFile(path,label=""):
    
    if label=="":
        label=" "
    else:
        label=" with"+label
    
    if os.path.isfile(path): 
        file=io.imread(path)  
        print("File "+path+label+" loaded...")
        return (file,True)
    else:
        print("File "+path+label+" do not exist...")
        return (file,False)
    

def accuracyTest(path_true,path_pred,result_path="results.txt",save_mode=1):
    print("------------------------------------------------")    
    true=openFile(path_true," ground truth")
    prediction=openFile(path_pred," prediction")
    
    if true[1]==False or prediction[1]==False:
        print("")
        return False
    
    true_t=unifyShape(true[0],prediction[0])
    prediction_t=unifyShape(prediction[0],true[0])
    

    true_f = true_t.flatten()
    pred_f = prediction_t.flatten()
    
    
    
    #dice=dice_coef(true_f/255,pred_f/255)
    #accuracy=accuracy_score(true_f,pred_f)
    #jaccard=jaccard_similarity_score(true_f,pred_f)
    
    score_w=Score(true_f,pred_f,0)
    score_b=Score(true_f,pred_f,255)
    score_b.calculate()
    score_w.calculate()
    
    prec_b=score_w.precision()
    prec_w=score_b.precision()
    
    #print("Dice coefficient:"+str(dice))
    #print("Accuracy score:"+str(accuracy))
    #print("Jaccard score:"+str(jaccard))
    
    prec_b=score_w.precision()
    prec_w=score_b.precision()
    print("Precision score for black:"+str(prec_b))
    print("Precision score for white:"+str(prec_w))
    
    prec_b=score_w.recall()
    prec_w=score_b.recall()
    print("Recall score for black:"+str(prec_b))
    print("Recall score for white:"+str(prec_w))
    
    prec_b=score_w.f1_score()
    prec_w=score_b.f1_score()
    print("F1 score for black:"+str(prec_b))
    print("F1 score for white:"+str(prec_w))
    
    prec_b=score_w.count
    prec_w=score_b.count
    pp=prec_b+prec_w
    print("Count for black:"+str(prec_b))
    print("Count for white:"+str(prec_w))
    print("Count :"+str(pp))
    print(pred_f.shape)
    
    print(classification_report(true_f,pred_f))
    
    
    
    if save_mode==1:
        results=open(result_path,'a')
        #results.write("Prediction:"+path_pred+";DC:"+str(dice)+";AS:"+str(accuracy)+";JS:"+str(jaccard)+"\n")
        results.close()
    print("------------------------------------------------")
   
    return True
        
if __name__=="__main__":
        f_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        
        for i in range(10,16):
        
            if i<10:
                data='DATA/mask/0'+str(i)+'_h.tif'

            else:
                data='DATA/mask/'+str(i)+'_h.tif'
            
            accuracyTest(data,'OUTPUT/128result_'+str(i)+'.jpg')

            
            for factor in f_ratio:
                accuracyTest(data,'OUTPUT/128result_'+str(i)+'_'+str(factor)+'_filtered.jpg')
                

        



    
    