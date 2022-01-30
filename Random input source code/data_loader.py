#Implementations for data simulation

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,LSTM
from keras.models import  Sequential
from keras.models import load_model

def get_mean_sd(para):
  mean=[]
  sd=[]
  #separate mean and std.
  for row in para:
    sp=row.split("+")
    mean.append(float(sp[0].strip().lstrip()))
    sd.append(float(sp[1].strip().rstrip()))
  return (np.asarray(mean),np.asarray(sd))

#Get (min, max) values using mean and std
def get_range(mean,var):
  out=[]
  for x,y in zip(mean,var):
      out.append((x-y,x+y))
  return out

#sample (n_samples) instances between min and max.
def sample_data(in_data,n_samples, is_d=False):
  t=n_samples
  n_samples=int(n_samples/41) #23 
  out_data=[]
  for row in in_data:
    step=(row[1]-row[0])/n_samples
    if(is_d==False):
      for i in np.arange(row[0],row[1]+step,step):
        out_data.append(i)
    else: #d decrease with increase in f
      for i in np.arange(row[0],row[1]+step,step)[::-1]:
        out_data.append(i)
  return out_data[:t]

#read F, D,Ds from dataset and increment the data.
def get_param_data(nsamples):
  data=pd.read_csv('data 2.csv')
  f=data['f%']
  ds=data['D* (10-3 mm2.Sec-1)']
  d=data['D (10-3 mm2.sec-1)']

  f_mean,f_sd=get_mean_sd(f)
  ds_mean,ds_sd=get_mean_sd(ds)
  d_mean,d_sd=get_mean_sd(d)

  #normalize the data
  f_mean/=100
  f_sd/=100
  ds_mean/=1000
  ds_sd/=1000
  d_mean/=1000
  d_sd/=1000
  
  n=100000   #total no of voxels considered

  #convert std to tolerance
  f_sd/=math.sqrt(n)
  d_sd/=math.sqrt(n)
  ds_sd/=math.sqrt(n)

  #obtain (min, max) pair
  f_pair=get_range(f_mean,f_sd)
  d_pair=get_range(d_mean,d_sd)
  ds_pair=get_range(ds_mean,ds_sd)

  #sample data between (min, max)
  f=sample_data(f_pair,int(nsamples))
  d=sample_data(d_pair,int(nsamples), is_d=True)
  ds=sample_data(ds_pair,int(nsamples))
  return (f,d,ds)

#Basic IVIM equation
def IVIM(b,F,D,Ds):
  return (F * np.exp( -1 * b * Ds ) + ( 1 - F )* np.exp( -1 * b * D ))

#Get b, S(b) values
def get_signal_data(b_vals,n_samples, prec=6):
  (f,d,ds)=get_param_data(n_samples)
  Sb=[]
  for (F,D,Ds) in zip(f,d,ds):
    Sb_=[]
    for b in b_vals:
      F=np.round(F,prec)
      D=np.round(D,prec)
      Ds=np.round(Ds,prec)
      Sb_.append(IVIM(b,F,D,Ds))
    Sb.append([np.asarray(Sb_), D, Ds, F])
  #Sb.append([np.asarray([0.0 for zr in b_vals]), 0.0, 0.0, 0.0]) #o data

  # print("Signal values",Sb[0])  
  return (np.asarray(b_vals),np.asarray(Sb))


#0,3,7,15: 9 or 5
def prepare_data(b_vals,Sb,n_steps,filt,t_b_i, max_b=900, factor=1):
    data=[]
    out=[]
    for x in Sb:
        row=[]
        for i in filt:
            row.append([(b_vals[i]*factor)/max_b,x[0][i]*factor, b_vals[t_b_i]/max_b])
        out.append([x[0][t_b_i]]) #100, 10, 10
        data.append(row)
    return (np.asarray(data),np.asarray(out))