import gc
import network as net

import DataExtractor as de
from skimage import io,external
import time
import scipy.misc


def predictOutput(image="DATA/mask/15_h.jpg",weights='model_w_new_512.h5',output_name='512result',channel=1,grey=False,dim=(512,512,1),f_ratio=[0.5]):
    IMAGE_PATH=image
    WEIGHTS_PATH=weights
    OUTPUT_PATH=output_name
    CHANNEL=channel
    GREY=grey

    model=net.NeuralNetwork()
    extractor=de.DataExtractor()
    
    model.load(dim,WEIGHTS_PATH)

    data=extractor.extractData(IMAGE_PATH,None,channel=CHANNEL,shape=dim,grey=GREY)
    #data=reframe(data,dim)
    shape=extractor.shape2(IMAGE_PATH,shape=dim)

    result = model.predictSet(data,shape,dim)

    io.imsave(OUTPUT_PATH+'.jpg',result)
    for ratio in f_ratio:
        im=scipy.misc.toimage(extractor.cutOff(result,ratio))
        im.save(OUTPUT_PATH+'_'+str(ratio)+'_filtered.png')

    gc.collect()

if __name__=="__main__":
    results=open('predictions.txt','a')
    
    for i in range(10,16):
        
        if i<10:
            data='DATA/mask/0'+str(i)+'_h.jpg'

        else:
            data='DATA/mask/'+str(i)+'_h.jpg'
        start=time.time()
        predictOutput(image=data,weights='model_w_new_128.h5',output_name=('OUTPUT/128result_'+str(i)),dim=(128,128,1),f_ratio=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        end=time.time()
        results.write("Model 128x128:"+str(end-start)+"\n")
        

    results.close()