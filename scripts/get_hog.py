#! /usr/bin/env python
# Python script to generate hog features from images.

import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
from skimage.feature import hog
from skimage.color import rgb2gray
import time

start_time=time.time()

cfg=yaml.load(open('../config.yaml','r'))
hdf5File='../dataset/'+cfg['defaults']['dataset']+'/dataset.hdf5'
data=h5py.File(hdf5File,'r')
_X_train_c=data['train_data']
_y_train_c=data['train_labels']
_X_test_c=data['test_data']
_y_test_c=data['test_labels']

hdf5file='../dataset/'+cfg['defaults']['dataset']+'/dataset_hog_9_8x8_3x3.hdf5'
f=h5py.File(hdf5file,'w')
f.create_dataset('train_labels',data=_y_train_c)
f.create_dataset('test_labels',data=_y_test_c)

#Training set HOG generations.
orientations=9
pixels_per_cell=(8, 8)
cells_per_block=(3, 3)

_hog_train=hog(rgb2gray(_X_train_c[0,:].reshape(32,32,3)),orientations,pixels_per_cell,cells_per_block)
train=f.create_dataset('train_data',(_X_train_c.shape[0],_hog_train.shape[0]))
train[0,:]=_hog_train
for i in range(1,_X_train_c.shape[0]):
    _hog_train=hog(rgb2gray(_X_train_c[i,:].reshape(32,32,3)),orientations,pixels_per_cell,cells_per_block)
    train[i,:]=_hog_train
    if(i%1000 == 0):
        print("%d images done"%(i))
print train.shape

_hog_test=hog(rgb2gray(_X_test_c[0,:].reshape(32,32,3)),orientations,pixels_per_cell,cells_per_block)
test=f.create_dataset('test_data',(_X_test_c.shape[0],_hog_test.shape[0]))
test[0,:]=_hog_train
for i in range(1,_X_test_c.shape[0]):
    _hog_test=hog(rgb2gray(_X_test_c[i,:].reshape(32,32,3)),orientations,pixels_per_cell,cells_per_block)
    test[i,:]=_hog_test
    if(i%1000 == 0):
        print("%d images done"%(i))
print test.shape


print "Total time: ", time.time()-start_time
