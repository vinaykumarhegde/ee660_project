#! /usr/bin/env python
# Python datasets file to match scikit-lear function call.
# Depends on config.yaml file.

import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
from skimage.feature import hog
from skimage.color import rgb2gray


class load_cifar(object):
    """create a cifar object dataset
    arg controls the amount of data to be returned"""
    cfg=yaml.load(open('../config.yaml','r'))
    hdf5File='../dataset/'+cfg['defaults']['dataset']+'/dataset.hdf5'
    data=h5py.File(hdf5File,'r')

    def __init__(self, train=0,test=0):
        self.test = test
        self.train = train
        self.X_train=np.array(self.data['train_data'],dtype='uint8')
        self.y_train=np.array(self.data['train_labels'],dtype='uint8')
        self.X_test=np.array(self.data['test_data'],dtype='uint8')
        self.y_test=np.array(self.data['test_labels'],dtype='uint8')
        if(train!=0 & train < self.y_train.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_train.shape[0],size=(train,))
            self.X_train=self.X_train[self.trIn,:]
            self.y_train=self.y_train[self.trIn]
        if(test!=0 & test < self.y_test.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_test.shape[0],size=(test,))
            self.X_test=self.X_test[self.trIn,:]
            self.y_test=self.y_test[self.trIn]


class load_cifar_gray(load_cifar):
    """ create a gray scale dataset
    -- inherits load_cifar.
    This is a direct matrix multiplication. This is
    much faster than individual image rgb-2-gray
    conversion.
    """
    rgb_weights=np.array([0.2125,0.7154,0.0721])
    def __init__(self, train=0,test=0):
        load_cifar.__init__(self,train,test)
        self.__rgb_weights = np.vstack(
            (self.rgb_weights[0]*np.eye(self.X_train.shape[1]/3),
            self.rgb_weights[1]*np.eye(self.X_train.shape[1]/3),
            self.rgb_weights[2]*np.eye(self.X_train.shape[1]/3)))
        self.X_train=self.X_train.dot(self.__rgb_weights)
        self.__rgb_weights = np.vstack(
            (self.rgb_weights[0]*np.eye(self.X_test.shape[1]/3),
            self.rgb_weights[1]*np.eye(self.X_test.shape[1]/3),
            self.rgb_weights[2]*np.eye(self.X_test.shape[1]/3)))
        self.X_test=self.X_test.dot(self.__rgb_weights)

class load_cifar_bow(object):
    """
    Imports bag of words from SIFT features. Current implementation
    uses 50 bins/words and histogram of SIFT features with following
    parameters:
    nOctaveLayers=4
    contrastThreshold=0.01
    edgeThreshold=20
    sigma=1.2
    Other parameters are also tested and found that several images
    will not have any SIFT features for those parameters.
    """
    __cfg=yaml.load(open('../config.yaml','r'))

    def __init__(self, train=0,test=0):
        self.__filename='../dataset/'
        self.__filename+=self.__cfg['defaults']['dataset']+'/dataset_sift_bag.hdf5'
        self.__data=h5py.File(self.__filename,'r')
        self.X_train=np.array(self.__data['train_data'])
        self.y_train=np.array(self.__data['train_labels'])
        self.X_test=np.array(self.__data['test_data'])
        self.y_test=np.array(self.__data['test_labels'])
        if(train!=0 & train < self.y_train.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_train.shape[0],size=(train,))
            self.X_train=self.X_train[self.trIn,:]
            self.y_train=self.y_train[self.trIn]
        if(test!=0 & test < self.y_test.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_test.shape[0],size=(test,))
            self.X_test=self.X_test[self.trIn,:]
            self.y_test=self.y_test[self.trIn]
        # Get finite elements:
        self._finiteTrain=np.isfinite(np.sum(self.X_train,axis=1))
        self.X_train=self.X_train[self._finiteTrain]
        self.y_train=self.y_train[self._finiteTrain]
        self._finiteTest=np.isfinite(np.sum(self.X_test,axis=1))
        self.X_test=self.X_test[self._finiteTest]
        self.y_test=self.y_test[self._finiteTest]


class load_cifar_hog(object):
    """
    Creates hog features on gray images.
    """
    __cfg=yaml.load(open('../config.yaml','r'))

    def __init__(self, train=0,test=0,orientations=9,pixels_per_cell=(8,8),cells_per_block=(3,3)):
        self.__filename='../dataset/'
        self.__filename+=self.__cfg['defaults']['dataset']+'/dataset_hog_'
        self.__filename+=str(orientations)+'_'
        self.__filename+=str(pixels_per_cell[0])+'x'+str(pixels_per_cell[1])+'_'
        self.__filename+=str(cells_per_block[0])+'x'+str(cells_per_block[1])+'.hdf5'
        self.__data=h5py.File(self.__filename,'r')
        self.X_train=np.array(self.__data['train_data'])
        self.y_train=np.array(self.__data['train_labels'])
        self.X_test=np.array(self.__data['test_data'])
        self.y_test=np.array(self.__data['test_labels'])
        if(train!=0 & train < self.y_train.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_train.shape[0],size=(train,))
            self.X_train=self.X_train[self.trIn,:]
            self.y_train=self.y_train[self.trIn]
        if(test!=0 & test < self.y_test.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_test.shape[0],size=(test,))
            self.X_test=self.X_test[self.trIn,:]
            self.y_test=self.y_test[self.trIn]


class load_rbm(object):
    """
    Creates rbm features on gray images.
    """
    __cfg=yaml.load(open('../config.yaml','r'))

    def __init__(self, train=0,test=0):
        self.__filename='../dataset/'
        self.__filename+=self.__cfg['defaults']['dataset']+'/dataset_rbm.hdf5'
        self.__data=h5py.File(self.__filename,'r')
        self.X_train=np.array(self.__data['train_data'])
        self.y_train=np.array(self.__data['train_labels'])
        self.X_test=np.array(self.__data['test_data'])
        self.y_test=np.array(self.__data['test_labels'])
        if(train!=0 & train < self.y_train.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_train.shape[0],size=(train,))
            self.X_train=self.X_train[self.trIn,:]
            self.y_train=self.y_train[self.trIn]
        if(test!=0 & test < self.y_test.shape[0]):
            self.trIn=np.random.random_integers(0,self.y_test.shape[0],size=(test,))
            self.X_test=self.X_test[self.trIn,:]
            self.y_test=self.y_test[self.trIn]



class sift_hog(object):
    '''
    This is to get a combined features from hog and sift bags.
    '''
    def __init__(self, train=0,test=0,):
        hog=load_cifar_hog(train,test)
        bow=load_cifar_bow(train,test)
        self.X_train=np.hstack((bow.X_train,hog.X_train[bow._finiteTrain]))
        self.y_train=bow.y_train
        self.X_test=np.hstack((bow.X_test,hog.X_test[bow._finiteTest]))
        self.y_test=bow.y_test



if __name__ == '__main__':
    print "load_cifar Example:"
    # c=load_cifar(100,100)
    # print c.X_train.shape,c.y_train.shape,c.X_test.shape,c.y_test.shape
    # d=load_cifar_gray(10,10)
    # print d.X_train.shape

    # Verify the images:
    # x=load_cifar().X_train[2,:].reshape([32,32,3],order='F')
    # x.transpose((1,0,2))
    # print load_cifar().y_train[2]
    # print x[0].shape
    # plt.imshow(x)
    # plt.show()
    # io.imshow(x)
    # io.show()

    # x=load_cifar_hog(10,10)
    # print x.X_train.shape
    # print x.X_test.shape

    # l=load_cifar().X_train[0,:].reshape(32,32,3,order='F')
    # ll=rgb2gray(l)
    # plt.imshow(ll)
    # plt.show()
    # print l.shape

    # x=load_cifar_bow()
    # print x.X_train.shape
    # print np.where(np.isnan(np.sum(x.X_train,axis=1))==True)

    # x=sift_hog(100,10)
    # print x.X_train.shape,x.y_train.shape,x.X_test.shape,x.y_test.shape
    x=load_rbm(100,10)
    print x.X_train.shape,x.X_test.shape
