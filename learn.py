import gc
import sys
import network as net
import DataExtractor as de
import numpy as np

def learnNetwork(image="DATA/mask/05_h.jpg",mask="DATA/mask/05_h.tif",mode='create',
                 old_weights='model_w_new_128.h5',new_weights='model_w_new_128.h5',epochs=1,
                 channel=1,gray=False,dim=(128,128,1)):
    
    IMAGE_PATH=image
    MASK_PATH=mask
    WEIGHTS_PATH=old_weights
    NEW_WEIGHTS_PATH=new_weights
    EPOCH_NUMBER=epochs
    CHANNEL=channel
    GREY=gray
    MODE=mode
   
    model=net.NeuralNetwork()
    extractor=de.DataExtractor()

    if MODE=='create':
        model.create(dim)
    elif MODE=='learn':
        model.load(dim,WEIGHTS_PATH)
    else:
        print('Bad mode. End of script...')
        sys.exit()

    train=extractor.extractData(IMAGE_PATH,MASK_PATH,channel=CHANNEL,shape=dim,grey=GREY)

    model.learn(train,epoch=EPOCH_NUMBER)
    model.save_weights(NEW_WEIGHTS_PATH)

    del train
    del model

    gc.collect()
	
if __name__=='__main__':
	FILES_NUMBER=int(sys.argv[2])
	MODE=sys.argv[1]
	EPOCHS=int(sys.argv[3])
	print(MODE)
	masks=np.array([])
	data=[]

	for i in range(1,FILES_NUMBER+1):
		if i<10:
			data=np.append(data,'DATA/mask/0'+str(i)+'_h.jpg')
			masks=np.append(masks,'DATA/mask/0'+str(i)+'_h.tif')
		else:
			data=np.append(data,'DATA/mask/'+str(i)+'_h.jpg')
			masks=np.append(masks,'DATA/mask/'+str(i)+'_h.tif')

	for i,img,maska in zip(range(len(data)),data,masks):
		print('Learning data:')
		print(img)
		print(maska)
    
		if MODE=='C' and i==0:
        
			learnNetwork(image=img,mask=maska,mode='create',epochs=1)
		else:
			learnNetwork(image=img,mask=maska,mode='learn',epochs=1)

