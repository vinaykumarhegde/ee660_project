#! /usr/bin/env python
# Create a sift bag of words
import numpy as np
import h5py
import yaml
import time
import cv2
from sklearn.cluster import KMeans


start_time=time.time()

#Declare number of bags
bags=50

cfg=yaml.load(open('../config.yaml','r'))
siftfile='../dataset/'+cfg['defaults']['dataset']+'/dataset_sift.hdf5'
datafile=h5py.File(siftfile,'r')
descs=datafile['train_data']
descs_t=datafile['test_data']
labels=datafile['train_lab']
labels_t=datafile['test_lab']
train_imgnos=datafile['train_imgno']
test_imgnos=datafile['test_imgno']

#     Basic initial variables for K-Means clustering. Using default from scikit image package.
max_iter=300;tol=1;
#     Perform the K-Means clustering and get the labels for training data
print 'K-Means fitting...'
kms=KMeans(n_clusters=bags,max_iter=max_iter,tol=tol).fit(descs)
#     Fit the test data and get labels for each feature.
print 'K-Means prediction'
test_labels=kms.predict(descs_t)
#     Format the X_train and X_test vectors.
X_train=np.zeros((int(np.amax(train_imgnos)+1),bags),dtype=float)
y_train=np.zeros((int(np.amax(train_imgnos)+1),),dtype='uint8')
for i in range(0,len(labels)):
    X_train[int(train_imgnos[i]),kms.labels_[i]]+=1
    if(not i%1000):
        print i,"images done."
#Normalize each rows.
X_train=(X_train.T/X_train.sum(axis=1)).T
for i,j in zip(train_imgnos,labels):
    y_train[int(i)]=int(j)
X_test=np.zeros((int(np.amax(test_imgnos)+1),bags),dtype=float)
y_test=np.zeros((int(np.amax(test_imgnos)+1),),dtype='uint8')
for i in range(0,len(labels_t)):
    X_test[int(test_imgnos[i]),test_labels[i]]+=1
    if(not i%1000):
        print i,"images done."
#Normalize each rows.
X_test=(X_test.T/X_test.sum(axis=1)).T
for i,j in zip(test_imgnos,labels_t):
    y_test[int(i)]=int(j)

outfile='../dataset/'+cfg['defaults']['dataset']+'/dataset_sift_bag.hdf5'
f=h5py.File(outfile,'w')
f.create_dataset('train_labels',data=y_train)
f.create_dataset('test_labels',data=y_test)
f.create_dataset('train_data',data=X_train)
f.create_dataset('test_data',data=X_test)
