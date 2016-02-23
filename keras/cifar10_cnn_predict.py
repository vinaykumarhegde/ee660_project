from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range


import numpy as np

'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(32, 32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 32, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64*8*8, 512, init='normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(512, nb_classes, init='normal'))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

weight_file='weights_epoch_35'
model.load_weights(weight_file)

datagen = ImageDataGenerator(
        featurewise_center=True, # set input mean to 0 over the dataset
        samplewise_center=False, # set each sample mean to 0
        featurewise_std_normalization=True, # divide inputs by std of the dataset
        samplewise_std_normalization=False, # divide each input by its std
        zca_whitening=False, # apply ZCA whitening
        rotation_range=20, # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2, # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2, # randomly shift images vertically (fraction of total height)
        horizontal_flip=True, # randomly flip images
        vertical_flip=False) # randomly flip images
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

print('Predicting train labels...')
(X_train2,y_train2)=[(X,y) for X,y in datagen.flow(X_train,y_train,batch_size=X_train.shape[0])][0]
y_pred_train=model.predict_classes(X_train2)
from sklearn.metrics import accuracy_score
print('Train accuracy:',accuracy_score(y_train,y_pred_train))

print('Predicting test labels...')
(X_test2,y_test2)=[(X,y) for X,y in datagen.flow(X_test,y_test,batch_size=X_test.shape[0])][0]
y_pred_test=model.predict_classes(X_test2)
print('Test accuracy:',accuracy_score(y_test,y_pred_test))

# Kaggle submission prodiction:
print('Predicting Kaggle test labels...')
import h5py
fo=h5py.File('kaggle_data.h5','r')
X_kag=fo['X_test']
y_kag=np.random.randint(0,10,(X_kag.shape[0],1))
(X_kag2,y_kag2)=[(X,y) for X,y in datagen.flow(X_kag,y_kag,batch_size=X_kag.shape[0])][0]
y_pred_kag=model.predict_classes(X_kag2)

#Make Kaggle submission csv file:
print('Creating Kaggle Submission csv file...')
import csv
classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
csv_dat=[['id','label']]
for i in range(len(y_pred_kag)):
   csv_dat.append([str(i+1),classes[y_pred_kag[i]]])
fcsv=open('kaggle_submission.csv','w')
a=csv.writer(fcsv,delimiter=',')
a.writerows(csv_dat)
fcsv.close()

#Save the model:
yml_model=model.to_yaml()
ff=open('model_architecture.yaml','w')
ff.write(yml_model)
ff.close()

# Save the predictions to HDF5 File:
f=h5py.File('model_predictions.h5','w')
f.create_dataset('X_train',data=X_train2)
f.create_dataset('y_train',data=y_train2)
f.create_dataset('y_pred_train',data=y_pred_train)
f.create_dataset('X_test',data=X_test2)
f.create_dataset('y_test',data=y_test2)
f.create_dataset('y_pred_test',data=y_pred_test)
f.create_dataset('y_kag',data=y_pred_kag)
