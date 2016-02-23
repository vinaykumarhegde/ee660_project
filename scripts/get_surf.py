#! /usr/bin/env python
# Python script to generate surf features and
# extract bag of features out of them.
# Stores the dictionary of surf features in hdf5 file
# this can be used later with knn clustering with
# different values of k.

import numpy as np
import h5py
import yaml
import time
import cv2

start_time=time.time()
surf=cv2.SURF(hessianThreshold=400, nOctaves=4, nOctaveLayers=2, extended=true, upright=false)

cfg=yaml.load(open('../config.yaml','r'))
hdf5File='../dataset/'+cfg['defaults']['dataset']+'/dataset.hdf5'
data=h5py.File(hdf5File,'r')
_X_train_c=data['train_data']
_y_train_c=data['train_labels']
_X_test_c=data['test_data']
_y_test_c=data['test_labels']

surffile='../dataset/'+cfg['defaults']['dataset']+'/dataset_surf.hdf5'
datfile=h5py.File(surffile,'w')

# Store Train Data:
count_train=0
train_data=datfile.create_dataset('train_data',(0,128),maxshape=(None,128)) #surf features.
train_imgno=datfile.create_dataset('train_imgno',(0,1),maxshape=(None,1)) #image numbers.
train_lab=datfile.create_dataset('train_lab',(0,1),maxshape=(None,1)) #labels for each features
for i,img in enumerate(_X_train_c):#Process every row in X_train
    img_gray=cv2.cvtColor(img.reshape(32,32,3,order='F'),cv2.COLOR_BGR2GRAY)
    _, des = surf.detectAndCompute(img_gray,None) #get the surf descriptors
    try:
        no_feats=des.shape[0] #number of features
    except AttributeError:
        print "Image", i, "has no descriptors. Ignoring this image."
        count_train+=1
        continue
    cur_index=train_data.shape[0]
    train_data.resize((cur_index+no_feats,128))
    train_imgno.resize((cur_index+no_feats,1))
    train_lab.resize((cur_index+no_feats,1))
    train_data[cur_index:,:]=des
    train_imgno[cur_index:,0]=i*np.ones((no_feats,),dtype='uint8')
    train_lab[cur_index:,0]=_y_train_c[i]*np.ones((no_feats,),dtype='uint8')

# Store test data:
count_test=0
test_data=datfile.create_dataset('test_data',(0,128),maxshape=(None,128)) #surf features.
test_imgno=datfile.create_dataset('test_imgno',(0,1),maxshape=(None,1)) #image numbers.
test_lab=datfile.create_dataset('test_lab',(0,1),maxshape=(None,1)) #labels for each features
for i,img in enumerate(_X_test_c):#Process every row in X_test
    img_gray=cv2.cvtColor(img.reshape(32,32,3,order='F'),cv2.COLOR_BGR2GRAY)
    _, des = surf.detectAndCompute(img_gray,None) #get the surf descriptors
    try:
        no_feats=des.shape[0] #number of features
    except AttributeError:
        print "Image", i, "has no descriptors. Ignoring this image."
        count_test+=1
        continue
    cur_index=test_data.shape[0] #Current index
    test_data.resize((cur_index+no_feats,128))
    test_imgno.resize((cur_index+no_feats,1))
    test_lab.resize((cur_index+no_feats,1))
    test_data[cur_index:,:]=des
    test_imgno[cur_index:,0]=i*np.ones((no_feats,),dtype='uint8')
    test_lab[cur_index:,0]=_y_train_c[i]*np.ones((no_feats,),dtype='uint8')

print "------- Statistics ---------"
print count_train, "train images don't have surf features"
print count_test, "test images don't have surf features"
print "Total time: ", time.time()-start_time
