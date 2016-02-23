#! /usr/bin/env python
# coding: utf-8

# In[1]:

#Linear classifiers test on original dataset.
import yaml
import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import cPickle
import time
from os.path import abspath


# In[2]:

#Read the configurations and dataset
with open('../config.yaml','r') as cfgFile:
    cfg=yaml.load(cfgFile)
hdf5File='../dataset/'+cfg['defaults']['dataset']+'/dataset.hdf5'
data=h5py.File(hdf5File,'r')
X_train=np.array(data['train_data'])
y_train=np.array(data['train_labels'])
X_test=np.array(data['test_data'])
y_test=np.array(data['test_labels'])


# In[3]:

C=1.0
classifiers = {'L1_logistic': LogisticRegression(C=C, penalty='l1'),
               'L2_logistic (OvR)': LogisticRegression(C=C, penalty='l2'),
               'L2_logistic_Multinomial': LogisticRegression(C=C, solver='lbfgs', multi_class='multinomial'),
               'Linear_SVC': SVC(kernel='linear', C=C, probability=True,random_state=0),
               'RBF_SVC': SVC(kernel='rbf', C=C),
               'kNN' : KNeighborsClassifier(n_neighbors=5, weights='uniform',algorithm='auto'),
              }


# In[4]:

for index,(name,classifier) in enumerate(classifiers.items()):
    start_time=time.time()
    print "Started ", name, " at: ", start_time
    classifier.fit(X_train,y_train)
    pklfile=open(name+'_learn.pkl','wb')
    print "Saving classifier file at: ",abspath(name+'_learn.pkl')
    cPickle.dump(classifier,pklfile)
    pklfile.close()
    print "Completed ",name, "classifier runs."
    print "Accuracy: ", classifier.score(X_test,y_test)
    print "Time taken: ", time.time()-start_time , "s"


# In[ ]:
