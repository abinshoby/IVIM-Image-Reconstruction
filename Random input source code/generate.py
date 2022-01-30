
from tarfile import LENGTH_NAME
from keras.engine.input_layer import Input
from keras.layers.core import Activation
from keras.losses import MSE, mean_squared_error
from keras.optimizer_v2.adam import Adam
from sklearn.utils import shuffle
from numpy.lib.financial import npv
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,LSTM,Bidirectional
from keras.models import  Sequential
from keras.models import load_model
from tensorflow.tools.docs.doc_controls import T
from data_loader import *
from keras import backend as K
import timeit
from skimage.metrics import structural_similarity as ssim

from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import adam_v2 as Adam
from lmfit import Model


np.random.seed(123)
K.set_floatx('float64') #use double precision float representation
img_res=256
prec=6 #output floating point precision

#definition of the model
def model(n_features):
    # import os
    # if(os.path.isdir('model')):
    #     print('Loading existing model')
    #     model=keras.models.load_model("model")
    #     return model
    # else:
        model=Sequential()
        model.add(Input(shape=[None, n_features], dtype=tf.float64, ragged=True))
        model.add(Bidirectional(LSTM(50,activation='relu')))
        model.add(Dense(1, activation='sigmoid')) #sigmoid for getting output in range 0-1
        opt=Adam.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)#beta_1=0.9, beta_2=0.999 for faster convergence and lr=0.0001 for better precision
        model.compile(optimizer=opt,loss='mse')
        return model

#stop the training when |val loss - train loss| + (val loss + train loss) is minimum
class EarlyStop(keras.callbacks.Callback):
    def __init__(self, value = 0, verbose = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.value = value
        self.verbose = verbose
        self.decrease_count=0
        self.prev_TT_TV=10000 #previous value of the metric


    def on_epoch_end(self, epoch, logs={}):
        curr_TT_TV=(logs['loss']+logs['val_loss'])+ abs(logs['loss']-logs['val_loss'])
        if(curr_TT_TV>self.prev_TT_TV and epoch>=30):
            self.decrease_count+=1
            if self.verbose >0:
                print("Epoch %05d: early stopping Threshold" % epoch)
            if(self.decrease_count>2):
                self.model.stop_training = True
        else:
            self.decrease_count=0
        self.prev_TT_TV=curr_TT_TV

def find_position(array, val):
    pos=-1
    for i in range(len(array)):
        if(array[i]>val):
            pos=i
            break
    if(pos==-1):
        return len(array)
    return pos

def save_simul_loss(filt, pred, b_vals, score):
    #append the score in csv
    import pandas as pd

    #convert indexes to its corresponding b-value
    b_val_subset=[b_vals[i] for i in filt]
    pred=b_vals[pred]

    df=pd.DataFrame({'subset':[str(b_val_subset)], 'target':[str(pred)], 'avg mse':[score]})
    # f=open('simulated_testing_loss.csv', 'a+')
    df.to_csv('results/simulated_testing_loss.csv', mode='a', index=False, header=False)

def calculate_ssim(result,actual):
    #no need to scale the image if we are giving data range parameter
    ssim_value = ssim(result, actual, data_range=actual.max() - actual.min())

    #in python the ssim range is -1 to 1 so scaling it to range 0-1
    return (ssim_value+1)/2
def compare(pred_i_dipy, pred, b_vals):
    pred_b=b_vals[pred_i_dipy]
    import dipy.data as r
    data, gtab=r.read_ivim()
    data=data.get_data()
    ROI_0=np.round(data[:,:,33,0],prec)
    data=np.load('datasets/dipy.npy')
    ROI=data[:, :, 33, :]
    ROI_actual=np.round(ROI[:,:,pred_i_dipy],prec)
    img_res=256
    for i in range(img_res):
        for j in range(img_res):
            ROI_actual[i][j]=ROI_actual[i][j]*ROI_0[i][j]
    ROI_actual=np.round(ROI_actual,prec)

    ROI_recon=np.round(np.asarray(pred).reshape(img_res, img_res),prec)
    for i in range(img_res):
        for j in range(img_res):
            ROI_recon[i][j]=ROI_recon[i][j]*ROI_0[i][j]
    ROI_recon=np.round(ROI_recon, prec)
    ssim=calculate_ssim(ROI_recon, ROI_actual)
    return ROI_recon, ROI_actual, ssim

def save_fig(name, data):
    # name=str(name)
    # print(name)
    # with open(name+'.npy', 'wb+') as p:
    #     np.save(p, data)
    
    fig3 = plt.figure(frameon=False)
    ax3 = plt.Axes(fig3, [0., 0., 1., 1.])
    ax3.set_axis_off()
    fig3.add_axes(ax3)
    ax3.imshow(data, aspect='auto', cmap='gray')
    fig3.savefig(name+".png")
    plt.close()    
def save_image_ssim(filt_dipy, pred_i_dipy, b_vals, recon_image, orig_image, ssim):
    subset=[b_vals[i] for i in filt_dipy]
    pred_b=b_vals[pred_i_dipy]

    #save images
    name='results/reconstructed/recon_'+str(subset)+'_'+str(pred_b)+'_ssim='+str(ssim)
    name_orig='results/original/orig_'+str(pred_b)
    save_fig(name, recon_image)
    save_fig(name_orig, orig_image)

    #save log to csv
    df=pd.DataFrame({'subset':[str(subset)], 'target':[str(pred_b)], 'ssim':[ssim], 'reconstructed image path':[name], 'original image path': [name_orig]})
    df.to_csv('results/dipy_testing_loss.csv', mode='a', index=False, header=False)

    


#generate the for given b-values
def generate(m,filt_dipy, pred_i_dipy, n_steps, dipy_data, Sb, b_vals, epochs=60, ndata=10000):
    # for pred_i_dipy in pred_order:
        
    
    x,y=prepare_data(b_vals,Sb,n_steps,filt_dipy,pred_i_dipy,max_b=max(b_vals))
    validation_split=int(0.4*ndata) #60-20-20 validation split
    testing_split=int(0.5*validation_split)

    print("data samples: "+str(len(x)))
    
    x,y=shuffle(x,y)

    #testing data 200
    tx=x[:testing_split]
    ty=y[:testing_split]

    # validation data 200
    vx=x[testing_split:validation_split]
    vy=y[testing_split:validation_split]

    #training data 600
    x=x[validation_split:]
    y=y[validation_split:]

    
    x=x.reshape(x.shape[0],x.shape[1],3)
    vx=vx.reshape(vx.shape[0],vx.shape[1],3)
    tx=tx.reshape(tx.shape[0],tx.shape[1],3)

    x=tf.ragged.constant(x)
    vx=tf.ragged.constant(vx)
    tx=tf.ragged.constant(tx)

    print("Training..")
    earlystop=EarlyStop()

    #start training
    m.fit(x,y,epochs=epochs,validation_data=(vx, vy),verbose=1, batch_size=16, callbacks=[earlystop] )

    #test on simulated data
    scores = m.evaluate(tx, ty, verbose=0)
    print("%s: %s" % ('loss', scores))
    save_simul_loss(filt_dipy, pred_i_dipy, b_vals, scores)

    #test on dipy
    arr_dipy=np.asarray(dipy_data)
    arr_dipy=arr_dipy.reshape(arr_dipy.shape[0],arr_dipy.shape[1],3)
    arr_dipy=tf.ragged.constant(arr_dipy)

    pred=m.predict(arr_dipy)
    pred=np.round(pred, prec) #predicted dipy image as a 1D array
    recon_image, orig_image, ssim=compare(pred_i_dipy, pred, b_vals)
    save_image_ssim(filt_dipy, pred_i_dipy, b_vals, recon_image, orig_image, ssim)
    return m

def biexpo(b,f,D,Ds):
    return((1-f)*np.exp(-b*D)+ f*np.exp(-b*Ds))


# to_be_done=[
#     [[0, 2, 4, 9, 13, 15, 16, 18], [19]],
#     [[0, 2, 4, 13, 15, 16, 18 ], [19]],
#     [[0, 2, 13, 16, 18 ], [19]],
#     [[0, 2, 13, 16 ], [19]]
# ]
# tests_input=[[0,20,140,200,1000],
# [0,20,160,300,1000],
# [0,40,180,300,900],
# [0,20,140,300,1000],
# [0,20,120,300,1000],
# [0,10,30,120,200,1000],
# [0,20,30,180,300,900],
# [0,20,80,180,600,1000],
# [0,10,80,180,300,900],
# [0,10,30,180,500,900],
# [0,10,40,80,120,200,500,1000],
# [0,10,30,80,160,180,500,1000],
# [0,20,30,80,120,180,300,1000],
# [0,20,30,40,120,180,200,1000],
# [0,40,180,900],
# [0,40,180,800],
# [0,80,200,1000]]



import dipy.data as r
import itertools

ndata=5000
_,gtab=r.read_ivim() #dipy dataset for testing
b_vals=gtab.bvals             
b_vals,Sb = get_signal_data(b_vals,ndata, prec=prec) #get simulated data
full_data=np.load('datasets/dipy.npy')


# test_input_index={int(b):ind for ind,b in enumerate(gtab.bvals)}
# for tt in tests_input:
#     for bt in gtab.bvals:
#         to_be_done.append([list(map(lambda x:test_input_index[x], tt)),[test_input_index[bt]]])
# print(to_be_done)

right=9
left=2 
trials=10
to_be_done=[]
from numpy.random import default_rng
rng = default_rng()

nb=len(b_vals)
range_b_orig=range(nb)
pos_500=nb-1
for i in range(nb):
    if(int(b_vals[i])==500):
        pos_500=i
        break
nb=pos_500+1


range_b=range(nb)
s=set(range_b)

for length in range(left, right):
    to_be_done=[]
    power_set=list(itertools.combinations(s, length))
    for _ in range(trials):
        rand_set_index=np.random.choice(len(power_set), size=1, replace=False)[0]
        rand_set=list(power_set[rand_set_index])
        for bt in list(range_b_orig):
            to_be_done.append([rand_set, [bt]])

    print('Total cases: '+str(len(to_be_done)))
    
    for cases in to_be_done:
            print(cases)
            m=model(3)
            filt=cases[0][:]
            predict_order=cases[1][0]
            n_steps=len(filt)
            rows=[] #dipy data

            for l in range(0,img_res):
                for t in range(0,img_res):
                    y=[]
                    for i in filt:
                        y.append([gtab.bvals[i]/max(b_vals),full_data[l,t,33,i], gtab.bvals[predict_order]/max(b_vals)])
                    rows.append(y)
            m=generate(m, filt, predict_order, n_steps, rows, Sb, b_vals, epochs=600, ndata=ndata)
            filt_b=[int(gtab.bvals[kb]) for kb in filt]
            target_b=int(gtab.bvals[predict_order])
            m.save('models/subset='+str(filt_b)+'_target='+str(target_b))

    
    




