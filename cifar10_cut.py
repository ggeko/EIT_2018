#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import os
import numpy as np

# import matplotlib.pyplot as plt
""" np.loadtxt can not read the FreeFEM output numerical csv file """


#############################################################################
# label size
#############################################################################
lsize = 32
#############################################################################
N  = 50000 * 3
ND = 10000 * 3
NT = 10000 * 3
#############################################################################
#  check float and int type to load correctly
#############################################################################
file_1 = 'cifar_data/raw_train_images_1.txt'
train_labels_1 = np.genfromtxt(file_1, comments='#',
                               delimiter=' ', dtype=np.float)
file_2 = 'cifar_data/raw_train_images_2.txt'
train_labels_2 = np.genfromtxt(file_2, comments='#',
                               delimiter=' ', dtype=np.float)
file_3 = 'cifar_data/raw_train_images_3.txt'
train_labels_3 = np.genfromtxt(file_3, comments='#',
                               delimiter=' ', dtype=np.float)
file_4 = 'cifar_data/raw_train_images_4.txt'
train_labels_4 = np.genfromtxt(file_4, comments='#',
                               delimiter=' ', dtype=np.float)
file_5 = 'cifar_data/raw_train_images_5.txt'
train_labels_5 = np.genfromtxt(file_5, comments='#',
                               delimiter=' ', dtype=np.float)
#############################################################################
file_t = 'cifar_data/raw_test_images.txt'
test_labels = np.genfromtxt(file_t, comments='#',
                            delimiter=' ', dtype=np.float)
#############################################################################
train_labels = np.zeros(N*lsize*lsize, dtype=np.float)
train_labels = train_labels.reshape( (N*lsize, lsize) )
train_labels[0:30000*lsize] = train_labels_1
train_labels[30000*lsize:60000*lsize] = train_labels_2
train_labels[60000*lsize:90000*lsize] = train_labels_3
train_labels[90000*lsize:120000*lsize] = train_labels_4
train_labels[120000*lsize:150000*lsize] = train_labels_5
train_labels = train_labels.reshape( (N, lsize, lsize) )
#############################################################################
test_labels = test_labels.reshape( (NT, lsize, lsize) )
#############################################################################
# make cut off circle data
#############################################################################
hsize = 32 // 2 ## 16
px = np.linspace(0.5-hsize, hsize-0.5, 2*hsize)
py = np.linspace(0.5-hsize, hsize-0.5, 2*hsize)
xx, yy = np.meshgrid(px,py)
xr = xx.ravel()
yr = yy.ravel()
d_pixel = np.sqrt(xr * xr + yr * yr)
in_image = (float(hsize) > d_pixel)
c_img = (in_image.astype(int)).reshape( (lsize, lsize) )
ci_train = np.array((c_img.tolist())*N, dtype=np.float)
ci_train = ci_train.reshape( (N, lsize, lsize) )
ci_test  = np.array((c_img.tolist())*NT, dtype=np.float)
ci_test = ci_test.reshape( (NT, lsize, lsize) )
#############################################################################
# cut off
#############################################################################
train_labels = train_labels * ci_train
test_labels = test_labels * ci_test
#############################################################################
# dvide data for save files
#############################################################################
images = []
images.append(train_labels[0:3*10000])
images.append(train_labels[3*10000:3*20000])
images.append(train_labels[3*20000:3*30000])
images.append(train_labels[3*30000:3*40000])
images.append(train_labels[3*40000:3*50000])
train_images = []
for fid in xrange(len(images)):
    train_images.append(np.asarray(images[fid], dtype=np.float))
    train_images[fid] = (train_images[fid]).reshape( (ND, lsize, lsize) )
# end for
#############################################################################
test_images = np.asarray(test_labels, dtype=np.float)
test_images = test_images.reshape( (NT, lsize, lsize) )
#############################################################################
# save the cutoff data
#############################################################################
for fid in xrange(len(train_images)):
    out_file = 'arrange/raw_train_images_cut_{}'.format(fid) + '.npy'
    np.save(out_file, train_images[fid])
# end for
out_file = 'arrange/raw_test_images_cut.npy'
np.save(out_file, test_images)
#############################################################################
