import os, sys
import Image
import numpy as np
import h5py

test_files=os.listdir('test/')
#test_files=test_files[0:5]

n_images=len(test_files)
f=h5py.File('kaggle_data.h5','w')
dat=f.create_dataset('X_test',(n_images,3,32,32),dtype='uint8')
for i in range(n_images):
   filepath='test/'+test_files[i]
   im=Image.open(filepath)
   r,g,b=im.split()
   fin=np.concatenate((r,g,b),axis=0).reshape(3,32,32)
   dat[i,:,:,:]=fin
   if(not i%1000):
      print i,'images done'

