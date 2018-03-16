#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import os
import numpy as np

# import matplotlib.pyplot as plt

""" np.loadtxt can not read the FreeFEM output numerical file """


##########################################################################
out_d = 'separate/'
##########################################################################



##########################################################################
# input size
##########################################################################
insize = 64
##########################################################################



##########################################################################
# separate period size
##########################################################################
s_period = [1, 2, 4]
s_size = [insize//s_period[0], insize//s_period[1], insize//s_period[2]]
##########################################################################



##########################################################################
# label size
##########################################################################
lsize = 32
##########################################################################



##########################################################################
channel = 2
NF = 10000 * 3
num_files = 5
N = NF * num_files
##########################################################################



"""
data_name = 'cifar_data/raw_train_output_arrange.npy'
x_list = np.load(data_name)
for j, pid in enumerate(s_period):
    #############################################################
    s_list = x_list[:, :, ::pid, ::pid]
    #############################################################
    out_file = 'x_train_sp{}x{}.npy'.format(s_size[j], s_size[j])
    np.save(out_d + out_file, s_list)
    #############################################################
# end for
"""



##########################################################################
# load input data
##########################################################################
x_list = []
data_name = 'arrange/raw_train_output_arrange_'
for i in xrange(num_files):
    file_name = data_name + '{}'.format(i) + '.npy'
    x_list.append(np.load(file_name))
# end for
##########################################################################
data_name = 'arrange/raw_test_output_arrange.npy'
x_test = np.load(data_name)
##########################################################################
# separate input data and save
##########################################################################
for j, pid in enumerate(s_period):
    #############################################################
    s_list  = []
    for i in xrange(num_files):
        s_list.append((x_list[i])[:, :, ::pid, ::pid])
    # end for
    s_test  = x_test[:, :, ::pid, ::pid]
    #############################################################
    for i in xrange(num_files):
        out_file = 'x_train_sp{}x{}_{}.npy'.format(s_size[j], s_size[j], i)
        np.save(out_d + out_file, s_list[i])
    # end for
    out_file = 'x_test_sp{}x{}'.format(s_size[j], s_size[j]) + '.npy'
    np.save(out_d + out_file, s_test)
    #############################################################
# end for
##########################################################################

