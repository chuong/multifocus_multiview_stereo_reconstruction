# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 11:40:15 2020

@author: u6265553
"""

'''This script generates the inverse transformation function required 
for fixed lens shift-scale calibration to align the synthetic images generated 
by Blender to a reference image to obtain fully in-focus image applying 
image stacking. this script calculates the shift and scaling 
required to align the images and generate required trabsformaton functions.'''

import glob
import numpy as np
import cv2
from skimage import restoration
from scipy.signal import convolve2d as conv2
from scipy import misc, optimize, special
from matplotlib import pylab as plt


pixelsize=4e-6
#focal length of the lens in meter
f_lens=65e-3
#focal length of the lens in pixel
f_lens_pixel=f_lens/pixelsize

forward_translation = 0.021
backward_translation = -0.019
average_focus_distance = 0.2

max_focus_distance = average_focus_distance + forward_translation
min_focus_distance= average_focus_distance + backward_translation

max_camera_focal_length = min_focus_distance*f_lens/(min_focus_distance-f_lens)
min_camera_focal_length = max_focus_distance*f_lens/(max_focus_distance-f_lens)

num_of_img=64
del_d = (max_camera_focal_length - min_camera_focal_length) / (num_of_img-1)   #in meter
del_d_pixel = del_d/pixelsize #in pixel
#focal length of the camera in pixel
f_camera_pixel = max_camera_focal_length/pixelsize

d_list=np.zeros([num_of_img])

for i in range(num_of_img):
    
    d_list[i]=f_camera_pixel-i*del_d_pixel

d_ref_pixel=d_list[0]

scaling=np.zeros([num_of_img])
shift_x=np.zeros([num_of_img])
shift_y=np.zeros([num_of_img])
H=np.zeros([num_of_img,3,3])

for i in range(num_of_img):
        
    #depth of the target calibration target
    d_target_pixel=d_list[i]
    
    scaling[i]=d_target_pixel/d_ref_pixel   
    shift_x[i]=(1-scaling[i])*(1080/2)
    shift_y[i]=(1-scaling[i])*(720/2)
    
    h=np.array([[scaling[i],0,shift_x[i]],[0,scaling[i],shift_y[i]],[0,0,1]])
    H[i]=(np.linalg.inv(h))

H_file = 'F:/Arif/fixed_lens/blender_images/insect3/Hs.npz'
homographies = []
np.savez(H_file, homographies=H)
    
