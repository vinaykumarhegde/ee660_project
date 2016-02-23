#! /bin/bash

#####################################################################
# Bash script to install basic tools for machine learning.
# This was intended to install tools in new AMI on AWS, but
# can also be usef for local machines.
# Author: Vinay (me@vnay.in)
#####################################################################

## Initial Setup.
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python-pip
sudo apt-get install python-dev

## Install BLAS and LAPACK for SciPy
sudo apt-get install libblas-dev
sudo apt-get install libblas-doc
sudo apt-get install liblapacke-dev
sudo apt-get install liblapack-doc
sudo apt-get install gfortran libopenblas-dev

## Install numpy and scipy
sudo pip install numpy==1.10.1
sudp pip install scipy==0.13.0

# Install Matplotlib dependencies
sudo apt-get install pkg-config
sudo apt-get install libpng-dev
sudo apt-get install libfreetype6-dev
sudo apt-get install libjpeg8-dev
sudo pip install matplotlib==1.5.0

## Install sklearn and skimage
sudo pip install scikit-image==0.11.3
sudo pip install scikit-learn==0.16.1

## Install IPython
sudo pip install ipython==4.0.0
sudo pip install ipyparallel
sudo pip install notebook==4.0.6
sudo pip install paramiko==1.16.0
sudo pip install sympy
sudo pip install nose==1.3.7
sudo pip install numexpr==2.4.6
sudo pip install pydot==1.0.2
sudo pip install pycrypto==2.6.1
sudo pip install virtualenv==13.1.2

## Install Pandas and Cython
sudo apt-get install libevent-dev
sudo pip install Cython==0.23.4
sudo pip install pandas==0.17.0

## Install Theano and Keras
sudo apt-get install git
sudo pip install Theano==0.7.0
sudo apt-get install libhdf5-dev
sudo pip install h5py
git clone https://github.com/fchollet/keras.git

## Setup Anaconda (Miniconda) - Incase you need it.
wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
chmod +x Miniconda-latest-Linux-x86_64.sh
./Miniconda-latest-Linux-x86_64.sh
source ~/.bashrc

## Install OpenCV 2.11.4
sudo apt-get -y install libopencv-dev build-essential cmake git \
  libgtk2.0-dev pkg-config libdc1394-22 x264 v4l-utils unzip\
  libdc1394-22-dev libjpeg-dev libpng12-dev libtiff4-dev libjasper-dev \
  libavcodec-dev libavformat-dev libswscale-dev libxine-dev \
  libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev \
  libv4l-dev libtbb-dev libqt4-dev libmp3lame-dev libopencore-amrnb-dev \
  libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev
cd~
wget https://github.com/Itseez/opencv/archive/2.4.11.zip -O opencv-2.4.11.zip
unzip opencv-2.4.11.zip
cd opencv-2.4.11
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON \
  -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON ..
make -j 4
sudo make install
cd ~
rm opencv-2.4.11.zip
sudo echo "/usr/local/lib/" >> /etc/ld.so.conf.d/opencv.conf
sudo echo \
  "PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig;exprt PKG_CONFIG_PATH" \
   >> ~/.bashsrc
sudo apt-get install python-opencv
sudo ln /dev/null /dev/raw1394


## Install Torch
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh
cd -

## Install Caffe -- Skipped the installation for this AMI
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install libatlas-base-dev
