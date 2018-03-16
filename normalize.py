#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import os
import glob
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import datetime
import math
import random
from six.moves import cPickle as pickle
import csv

# import matplotlib
# import matplotlib.pyplot as plt


""" np.loadtxt can not read the FreeFEM output numerical file """

"""
def calc_standard(idata):
    # calc standardization data
    nn = len(idata)
    data_shape = idata.shape
    idata = (idata.reshape( (nn, -1) )).copy()
    l_mean = idata.mean(axis=0)
    idata -= l_mean
    l_std = idata.std(axis=0)
    return l_mean, l_std



def sdd_by_data(idata, l_mean, l_std):
    # Standardization (Local Contrast Normalization) 
    # for input data (channel=2)
    nn = len(idata)
    data_shape = idata.shape
    idata = idata.reshape( (nn, -1) )
    idata -= l_mean
    idata /= (l_std + 1e-6)
    return idata.reshape( data_shape )



def whiten(idata):
    # ZCA Whitening for input data (channel=2)
    nn = len(idata)
    idata = idata.reshape( (nn, channel*insize*insize) )
    mean = np.mean(idata, axis=0)
    mdata = idata - mean
    sigma = np.dot(mdata.T, mdata) / mdata.shape[0]
    U, S, V = linalg.svd(sigma)
    components = np.dot(np.dot(U, np.diag(1 / np.sqrt(S + 1e-6))), U.T)
    wdata = np.dot(mdata, components.T)
    return components, mean, wdata.reshape( (nn, channel, insize ,insize) )

"""

def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data



def rgb_split(c_img):
    imgs = []
    for i in xrange(3):
        imgs.append(c_img[i])
    # end for    
    imgs = np.asarray(imgs, dtype=np.float)
    return imgs


#############################################################################
# load CIFAR-10 training dataset
# transform images ( color image -> RGB separate 3 images )
#############################################################################
train_color_images = np.zeros((50000, 3 * 32 * 32), dtype=np.int)
train_file ='chainer-cifar10/cifar-10-batches-py/data_batch*'
for i, data_fn in enumerate(sorted(glob.glob(train_file))):
    batch = unpickle(data_fn)
    train_color_images[i * 10000:(i + 1) * 10000] = batch['data']
# end for
train_color_images = np.asarray(train_color_images, dtype=np.int)
train_color_images = train_color_images.reshape((50000, 3, 32, 32))
images = []
for c_img in train_color_images:
    rgb_imgs = rgb_split(c_img)
    for img in rgb_imgs:
        images.append(img)
    # end for
# end for
train_images = np.asarray(images, dtype=np.float)
train_images = train_images.reshape((50000 * 3, 32, 32))
#############################################################################
# load CIFAR-10 test dataset
# transform images ( color image -> RGB separate 3 images )
#############################################################################
test = unpickle('chainer-cifar10/cifar-10-batches-py/test_batch')
test_color_images = np.asarray(test['data'], dtype=np.float)
test_color_images = test_color_images.reshape((10000, 3, 32, 32))
images = []
for c_img in test_color_images:
    rgb_imgs = rgb_split(c_img)
    for img in rgb_imgs:
        images.append(img)
    # end for
# end for
test_images = np.asarray(images, dtype=np.float).reshape((10000 * 3, 32, 32))
#############################################################################
# Normalize for image data
#############################################################################
train_images /= 255.0
test_images  /= 255.0
#############################################################################
images_1 = train_images[0:3*10000]
images_2 = train_images[3*10000:3*20000]
images_3 = train_images[3*20000:3*30000]
images_4 = train_images[3*30000:3*40000]
images_5 = train_images[3*40000:3*50000]
train_images_1 = np.asarray(images_1, dtype=np.float)
train_images_1 = train_images_1.reshape( (3*10000*32, 32) )
train_images_2 = np.asarray(images_2, dtype=np.float)
train_images_2 = train_images_2.reshape( (3*10000*32, 32) )
train_images_3 = np.asarray(images_3, dtype=np.float)
train_images_3 = train_images_3.reshape( (3*10000*32, 32) )
train_images_4 = np.asarray(images_4, dtype=np.float)
train_images_4 = train_images_4.reshape( (3*10000*32, 32) )
train_images_5 = np.asarray(images_5, dtype=np.float)
train_images_5 = train_images_5.reshape( (3*10000*32, 32) )
#############################################################################
test_images = np.asarray(test_images, dtype=np.float)
test_images = test_images.reshape( (3*10000*32, 32) )
#############################################################################
# save the transformed data
#############################################################################
with open('cifar_data/raw_train_images_1.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(train_images_1.tolist())

with open('cifar_data/raw_train_images_2.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(train_images_2.tolist())

with open('cifar_data/raw_train_images_3.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(train_images_3.tolist())

with open('cifar_data/raw_train_images_4.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(train_images_4.tolist())

with open('cifar_data/raw_train_images_5.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(train_images_5.tolist())

with open('cifar_data/raw_test_images.txt', 'w') as f:
    writer = csv.writer(f, delimiter=' ')
    writer.writerows(test_images.tolist())

#############################################################################
