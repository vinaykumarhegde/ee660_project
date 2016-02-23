#! /usr/bin/env python
#Linear classifiers test on original dataset.
import yaml
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import time

classifiers = {'L2 logistic (OvR)': LogisticRegression(C=1, penalty='l2')}
#Read the configurations and dataset
with open('../config.yaml','r') as cfgFile:
    cfg=yaml.load(cfgFile)
hdf5File='../dataset/'+cfg['defaults']['dataset']+'/dataset.hdf5'
data=h5py.File(hdf5File,'r')
X_train=np.array(data['train_data'])
y_train=np.array(data['train_labels'])
X_test=np.array(data['test_data'])
y_test=np.array(data['test_labels'])

start_time=time.time()

for index,(name,classifier) in enumerate(classifiers.items()):
    print index, name, classifier.fit(X_train,y_train).score(X_test,y_test)

print("Execution time: %s seconds" %(time.time()-start_time) )
