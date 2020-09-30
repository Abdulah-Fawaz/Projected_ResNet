#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 10:29:29 2020

@author: fa19
"""

import numpy as np

rotation_arr = np.load('/home/fa19/Documents/my_version_spherical_unet/rotations_array.npy').astype(int)


import nibabel as nb
import sys 

xy_points = np.load('ico_6_xy_points.npy')
grid = np.load('grid_170_square.npy')

sys.path.append( "/home/fa19/Downloads/icosahedrons")

surface_ico = nb.load('/home/fa19/Downloads/icosahedrons/ico-6.surf.gii')

coords = surface_ico.darrays[0].data / 100

faces = surface_ico.darrays[1].data



chosen_im = nb.load('/home/fa19/Documents/dHCP_Data/Raw/myelin/CC00053XX04-8607-left_myelin_map.shape.gii')

chosen_im = chosen_im.darrays[0].data
chosen_im = np.reshape(chosen_im, [chosen_im.shape[0], 1])
im = chosen_im




template = nb.load('/home/fa19/Documents/dHCP_Data/Raw/myelin/CC00053XX04-8607-left_myelin_map.shape.gii')
    
template.darrays[0].data = im
nb.save(template, 'testing_rotations/original.shape.gii')
    
#for rot in range(len(rotation_arr)):
#    
#    template.darrays[0].data = im[rotation_arr[rot]]
#    
#    nb.save(template, 'testing_rotations/rotated' + str(rot)+'.shape.gii')
im = im[rotation_arr[20]]   
import time
start = time.time()
image1 = griddata(xy_points, im, grid, 'nearest')
end = time.time()

print(end-start)
import matplotlib.pyplot as plt
plt.matshow(image1.reshape(170,170))

import time
start = time.time()
image2 = griddata(xy_points, im, grid, 'linear')
end = time.time()
plt.matshow(image2.reshape(170,170))

print(end-start)
