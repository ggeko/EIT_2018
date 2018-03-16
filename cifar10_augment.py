#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import os
import numpy as np

import math

from scipy import linalg


import cPickle as pickle

# import matplotlib.pyplot as plt

""" np.loadtxt can not read the FreeFEM output numerical file """


##########################################################################
in_d  = 'separate/'
out_d = 'data/'
##########################################################################
# input size
##########################################################################
data_state = '32x32'
insize = 32
##########################################################################
## data_state = '64x64'
## insize = 64
##########################################################################
# label size
##########################################################################
lsize = 32
##########################################################################
# file size
##########################################################################
channel = 2
NF = 10000 * 3
num_files = 5
# num_files = 1
N = NF * num_files
##########################################################################
new_files = 5
##########################################################################
## size of valid data
##########################################################################
NV = 5000
##########################################################################
##########################################################################
def calc_standard(idata):
    # calc standardization data
    nn = len(idata)
    data_shape = idata.shape
    idata = (idata.reshape( (nn, -1) )).copy()
    l_mean = idata.mean(axis=0)
    idata -= l_mean
    l_std = idata.std(axis=0)
    return nn, l_mean, l_std
##########################################################################
def calc_lcn_data(num_arr, mean_arr, std_arr):
    len_arr = len(num_arr)
    num_array = np.array(num_arr, dtype = np.int).reshape((len_arr, 1))
    mean_array = np.array(mean_arr, dtype = np.float)
    std_array = np.array(std_arr, dtype = np.float)
    NA = num_array.sum()
    l_mean = (num_array * mean_array).sum(axis = 0) / float(NA)
    sq_mean = mean_array * mean_array
    sq_std = std_array * std_array
    sq_l_std = ((((sq_mean + sq_std) * num_array).sum(axis = 0) / float(NA)) - 
               (l_mean * l_mean))
    l_std = np.sqrt(sq_l_std)
    lcn = {}
    lcn['number'] = NA
    lcn['mean']   = l_mean
    lcn['std']    = l_std
    lcn_dir = 'LCN_{}'.format(data_state)
    if not os.path.exists(lcn_dir):
        os.makedirs(lcn_dir)
    # end if
    l_name = lcn_dir + '/lcn.pkl'
    with open(l_name, mode='wb') as f:
        pickle.dump(lcn, f, protocol=2)
    # end with
##########################################################################
def load_lcn_data():
    lcn_dir = 'LCN_{}'.format(data_state)
    l_name = lcn_dir + '/lcn.pkl'
    with open(l_name, mode='rb') as f:
        lcn_data = pickle.load(f)
    # end with
    l_mean = lcn_data['mean']
    l_std  = lcn_data['std']
    return l_mean, l_std
##########################################################################
def sdd_by_data(idata, l_mean, l_std):
    # Standardization (Local Contrast Normalization) 
    # for input data (channel=2)
    nn = len(idata)
    data_shape = idata.shape
    idata = idata.reshape( (nn, -1) )
    idata -= l_mean
    idata /= (l_std + 1e-6)
    return idata.reshape( data_shape )
##########################################################################
def save_final_data(x_arr, y_arr, fid):
    out_dir = out_d + 'data_{}/'.format(data_state)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # end if
    out_file = out_dir + 'x_train_' + str(fid) + '.npy'
    np.save(out_file, x_arr)
    out_file = out_dir + 'y_train_' + str(fid) + '.npy'
    np.save(out_file, y_arr)
##########################################################################
def save_test_data(x_val, x_tes, y_val, y_tes):
    out_dir = out_d + 'data_{}/'.format(data_state)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # end if
    #########################################
    out_file = out_dir + 'x_valid.npy'
    np.save(out_file, x_val)
    #########################################
    out_file = out_dir + 'x_test.npy'
    np.save(out_file, x_tes)
    #########################################
    out_file = out_dir + 'y_valid.npy'
    np.save(out_file, y_val)
    #########################################
    out_file = out_dir + 'y_test.npy'
    np.save(out_file, y_tes)
    #########################################
##########################################################################
##########################################################################
def data_roll_right(d_array, s_num): ## s_num = 0,1,2,3 (random)
    d_shift = s_num * insize / 4  ## 右 90 * s_num 度回転用 d_shift = 32/4
    tmp = np.roll(d_array, - d_shift, axis = 2)
    return np.roll(tmp, - d_shift, axis = 3)
##########################################################################
##########################################################################
def data_fliplr(d_array): ## 裏返し
    return d_array[:,:,::-1,::-1]
##########################################################################
##########################################################################
# for train data (input data and labels data)
##########################################################################
##########################################################################
# load train data
##########################################################################
num_list  = []
mean_list = []
std_list  = []
##########################################################################
for fid in xrange(num_files):
    ######################################################################
    print('load train data:', fid)
    data_name = 'x_train_sp{}_{}.npy'.format(data_state, fid)
    tmp_data = np.load(in_d + data_name)
    print('shape of data {}:'.format(fid), tmp_data.shape)
    data = tmp_data.astype(np.float)
    data = data.reshape( (NF, channel, insize, insize) )
    ######################################################################
    print('load label data:', fid)
    label_name = 'arrange/raw_train_images_cut_{}'.format(fid) + '.npy'
    tmp_labels = np.load(label_name)
    print('shape of labels {}:'.format(fid), tmp_labels.shape)
    labels = tmp_labels.astype(np.float)
    labels = labels.reshape( (NF, 1, lsize, lsize) )
    ######################################################################
    ######################################################################
    for s_num in xrange(4):
        #############################################################
        new_data = data_roll_right(data, s_num)
        n_val, m_val, s_val = calc_standard(new_data)
        num_list.append(n_val)
        mean_list.append(m_val)
        std_list.append(s_val)
        print('n_val =', n_val, 'm_val =', m_val, 's_val =', s_val)
        #############################################################
        new_data = data_fliplr(new_data)
        n_val, m_val, s_val = calc_standard(new_data)
        num_list.append(n_val)
        mean_list.append(m_val)
        std_list.append(s_val)
        print('n_val =', n_val, 'm_val =', m_val, 's_val =', s_val)
        #############################################################
    # end for
    ######################################################################
    ######################################################################
# end for
##########################################################################
##########################################################################
# calc for lcn mean and std
##########################################################################
print('calc the mean and std for standardization')
calc_lcn_data(num_list, mean_list, std_list)
##########################################################################
lcn_mean, lcn_std = load_lcn_data()
##########################################################################
for fid in xrange(num_files):
    ######################################################################
    print('load train data:', fid)
    data_name = 'x_train_sp{}_{}.npy'.format(data_state, fid)
    tmp_data = np.load(in_d + data_name)
    print('shape of data {}:'.format(fid), tmp_data.shape)
    data = tmp_data.astype(np.float)
    data = data.reshape( (NF, channel, insize, insize) )
    ######################################################################
    print('load label data:', fid)
    label_name = 'arrange/raw_train_images_cut_{}'.format(fid) + '.npy'
    tmp_labels = np.load(label_name)
    print('shape of labels {}:'.format(fid), tmp_labels.shape)
    labels = tmp_labels.astype(np.float)
    labels = labels.reshape( (NF, 1, lsize, lsize) )
    ######################################################################
    print('standardization of train input data', fid)
    data = sdd_by_data(data, lcn_mean, lcn_std)
    x_train = np.asarray(data, dtype=np.float32)
    x_train = x_train.reshape( (-1, 2, insize, insize) )
    y_train = np.asarray(labels, dtype=np.float32)
    y_train = y_train.reshape( (-1, 1, lsize, lsize) )
    #######################################################
    print('final train data:', fid, ', shape:', x_train.shape)
    print('final label data:', fid, ', shape:', y_train.shape)
    #######################################################
    print('save the final data')
    save_final_data(x_train, y_train, fid)
    ######################################################################
# end for
##########################################################################
##########################################################################

##########################################################################
##########################################################################
# for test data (input data and label data)
##########################################################################
##########################################################################

##########################################################################
# load test data
##########################################################################
print('load test data')
data_name = 'x_test_sp{}.npy'.format(data_state)
data = np.load(in_d + data_name)
print('test data  shape :', data.shape, data.dtype)
##########################################################################
label_name = 'arrange/raw_test_images_cut.npy'
labels = np.load(label_name)
print('test label shape :', labels.shape, labels.dtype)
##########################################################################
data   = np.asarray(data, dtype=np.float)
data   = data.reshape( (NF, 2, insize, insize) )
labels = np.asarray(labels, dtype=np.float)
labels = labels.reshape( (NF, 1, lsize, lsize) )
##########################################################################
print('test data  shape :', data.shape)
print('test label shape :', labels.shape)
##########################################################################
# Standardization for input test data
##########################################################################
print('standardization of test input data')
lcn_mean, lcn_std = load_lcn_data()
data = sdd_by_data(data, lcn_mean, lcn_std)
##########################################################################
print('trans to numpy and reshape for test data')
test_data = np.asarray(data, dtype=np.float)
test_data = test_data.reshape( (NF, 2, insize, insize) )
print('standardized test data   : shape :', test_data.shape)
##########################################################################
test_labels = np.asarray(labels, dtype=np.float)
test_labels = test_labels.reshape( (NF, 1, lsize, lsize) )
print('standardized test labels : shape :', test_labels.shape)
##########################################################################
# split valid and test data
##########################################################################
print('split into valid and test data')
x_valid, x_test = np.split(test_data,   [NV])
y_valid, y_test = np.split(test_labels, [NV])
x_valid = np.asarray(x_valid, dtype=np.float32)
y_valid = np.asarray(y_valid, dtype=np.float32)
x_test  = np.asarray(x_test, dtype=np.float32)
y_test  = np.asarray(y_test, dtype=np.float32)
#######################################################
print('final train valid data : shape :', x_valid.shape)
print('final train test  data : shape :', x_test.shape)
print('final label valid data : shape :', y_valid.shape)
print('final label test  data : shape :', y_test.shape)
##########################################################################
# save valid and test data
##########################################################################
print('save the split valid and test data')
save_test_data(x_valid, x_test, y_valid, y_test)
##########################################################################
##########################################################################
