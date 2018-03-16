#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function

import sys
import os
import numpy as np

# import matplotlib.pyplot as plt

""" np.loadtxt can not read the FreeFEM output numerical file """


##########################################################################
insize = 64
##########################################################################

channel = 2

NF = 10000 * 3

num_files = 5

##########################################################################
##########################################################################
# for train data
##########################################################################
##########################################################################
for fid in xrange(num_files):
    print('file load:', fid)
    f_name = 'cifar_data/raw_train_output_{}.txt'.format(fid + 1)
    t_data = np.genfromtxt(f_name, comments='#',delimiter=' ', dtype=np.float)
    print('data shape:', t_data.shape)
    predata = t_data.reshape( (NF, insize*channel, insize) )
    ######################################################################
    # preperation for the correct sequence data
    ######################################################################
    print('arrange order for data:', fid) 
    tmp = np.zeros(channel*insize*insize, dtype=np.float)
    tmp = tmp.reshape((channel, insize, insize))   
    data = np.zeros(NF*channel*insize*insize, dtype=np.float)
    data = data.reshape((NF, channel, insize, insize))
    ######################################################################
    ## arrangement for order
    ######################################################################
    prelist = predata.tolist()
    for (i, inputs) in enumerate(prelist):
        for (j, d_list) in enumerate(inputs):
            index = j // 2
            if (j % 2) == 0:
                tmp[0, index] = d_list
            elif (j % 2) == 1:
                tmp[1, index] = d_list
            # end if
        # end for
        data[i] = tmp
    # end for
    ######################################################################
    ######################################################################
    # compensate the diff of electrode position and measure position
    ######################################################################
    data = np.array(data, dtype= np.float)
    data = data.reshape((NF, channel, insize, insize))
    data = data[:,:,:,::-1]
    data = np.roll(data, - insize/4, axis=3)
    ######################################################################
    ######################################################################
    # save data
    ######################################################################
    print('save data:', fid)
    out_file = 'arrange/raw_train_output_arrange_{}'.format(fid) + '.npy'
    np.save(out_file, data)
    ######################################################################
# end for
##########################################################################
##########################################################################


##########################################################################
##########################################################################
# for test data
##########################################################################
##########################################################################
print('load test data')
file_name = 'cifar_data/raw_test_output.txt'
test_data = np.genfromtxt(file_name, comments='#',
                          delimiter=' ', dtype=np.float)
##########################################################################
predata = np.zeros(NF*channel*insize*insize, 
                   dtype=np.float).reshape((NF*channel*insize, insize))
predata = test_data
##########################################################################
# data correct sequence
##########################################################################
print('arrange order for test data') 
predata = predata.reshape( (NF, insize*channel, insize) )
tmp = np.zeros(channel*insize*insize, 
               dtype=np.float).reshape((channel, insize, insize))   
data = np.zeros(NF*channel*insize*insize, 
                dtype=np.float).reshape((NF, channel, insize, insize))
##########################################################################
prelist = predata.tolist()
for (i, inputs) in enumerate(prelist):
    for (j, d_list) in enumerate(inputs):
        index = j // 2
        if (j % 2) == 0:
            tmp[0, index] = d_list
        elif (j % 2) == 1:
            tmp[1, index] = d_list
        # end if
    # end for
    data[i] = tmp
# end for
##########################################################################
test_data = data.reshape( (NF, channel, insize, insize) )
##########################################################################
##########################################################################
# compensate the diff of electrode position movement and measure position
##########################################################################
test_data = test_data[:,:,:,::-1]
test_data = np.roll(test_data, - insize/4, axis=3)
##########################################################################
##########################################################################
# save the arrange test data
##########################################################################
print('save test data')
out_file = 'arrange/raw_test_output_arrange.npy'
np.save(out_file, test_data)
##########################################################################
