#! /usr/bin/env python
'''Python script to read the CIFAR-10 Dataset and store as
HDF5 file. This is to increase the inter operability.

depends on PROJ_PATH environment variable.
'''
import os
import h5py
import sys
import yaml
import numpy as np
import time


def unpickle(file):
    import cPickle
    with open(file,'r') as fo:
        dict=cPickle.load(fo)
    fo.close()
    return dict

def main(argv,defaults):
    '''Read the cPickle values in the CIFAR-10 Dataset and convert it to
    HDF5 table'''
    comb_data=np.empty([1,3072],dtype='uint8')
    comb_label=np.empty([1,],dtype='uint8')
    for i in range(1,6):
        filename='../dataset/'+defaults['dataset']+'/data_batch_'+str(i)
        x=unpickle(filename)
        comb_data=np.concatenate((comb_data,x['data']),axis=0)
        comb_label=np.concatenate((comb_label,np.array(x['labels'])),axis=0)
    test_file='../dataset/'+defaults['dataset']+'/test_batch'
    x=unpickle(test_file)
    hdf5file='../dataset/'+defaults['dataset']+'/dataset.hdf5'
    with h5py.File(hdf5file,'w') as f:
        f.create_dataset('train_data',data=comb_data[1:])
        f.create_dataset('train_labels',data=comb_label[1:])
        f.create_dataset('test_data',data=x['data'])
        f.create_dataset('test_labels',data=np.array(x['labels']))


if __name__ == '__main__':
    start_time=time.time()
    with open('../config.yaml','r') as yamlfile:
        cfg=yaml.load(yamlfile)
    main(sys.argv,cfg['defaults'])
    print("Execution time: %s seconds" %(time.time()-start_time) )
