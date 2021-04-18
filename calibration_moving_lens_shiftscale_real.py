# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:17:44 2020

@author: Arif
"""
'''This script generates the inverse transformation function required 
for moving lens shift-scale calibration to align the images to a reference 
image to obtain fully in-focus image applying image stacking. At first, this 
script computes magnification. Then it computes depth of the target from the 
lens and the focal length of the camera (distance between the lens and the 
sensor plane) using the magnification. It calculates the shift and scaling 
required to align the images and generate required trabsformaton functions.'''

import glob
import numpy as np
import cv2
from skimage import restoration
from scipy.signal import convolve2d as conv2
from scipy import misc, optimize, special
from matplotlib import pylab as plt


'''
#computing magnification

imglist=glob.glob('F:/Arif/moving_lens/moving_lens_calibration/calibration_images/*.jpg')
img1 = cv2.imread(imglist[0])
height,width,channel=np.shape(img1)


#resizing the image
img=cv2.resize(img1,(int(width/4),int(height/4)))

# converting to gray
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# reducing the out of focus (blurring) effect
psf = np.ones([5, 5]) / 25
image = conv2(gray, psf, 'same')
image += 0.1 * image.std() * np.random.standard_normal(image.shape)
deconvolved = restoration.wiener(image, psf, 1, clip=False)

# Find the circle centers
ret, centers = cv2.findCirclesGrid(np.uint8(deconvolved), (5,5), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)

# If found, image points
if ret==True:
    
    #imgpoints.append(centers)
    cv2.drawChessboardCorners(img1, (5,5), centers*4, ret)
    
#cv2.imshow('image',img1)
#cv2.imwrite('F:/Arif/moving_lens/calibration/trans_img/img.png',img1)  

#magnification
pixelsize=4e-6
#physical distance between the dots in the calibration target obtained from measurement 
#in mm
Del_Y=1.7e-3
#converting Del_Y in pixel  
Del_Y_pixel=Del_Y/pixelsize 
centers1=centers*4 

#pixel distance between the dots in the calibration target image 
del_y_pixel=np.max(np.array([(centers1[1][0][0]-centers1[0][0][0]),(centers1[2][0][0]-centers1[1][0][0]),(centers1[3][0][0]-centers1[2][0][0])]))
#computing magnification
M=del_y_pixel/Del_Y_pixel
'''

# Magnification for moving lens capture
M=1.3
#M=round(M)
pixelsize=4e-6
#focal length of the lens in meter
f_lens=65e-3
#focal length of the lens in pixel
f_lens_pixel=f_lens/pixelsize

#focal length of the camera in pixel
f_camera_pixel=f_lens_pixel*((M)+1)
#depth of the current calibration target in pixel
d_pixel=f_camera_pixel/(M)
#linear displacement of the camera
del_d=0.25e-3     #in meter
del_d_pixel=del_d/pixelsize #in pixel

#num_of_img=np.size(imglist)
num_of_img=61
d_list=np.zeros([num_of_img])

for i in range(num_of_img):
            
    d_list[i]=d_pixel-i*del_d_pixel

d_ref_pixel=d_list[63]

#misalignment coefficient
#sigma=0.028
#gamma=-0.006

scaling=np.zeros([num_of_img])
shift_x=np.zeros([num_of_img])
shift_y=np.zeros([num_of_img])
H=np.zeros([num_of_img,3,3])

for i in range(num_of_img):
           
    #depth of the target calibration target
    d_target_pixel=d_list[i]
    
    scaling[i]=d_ref_pixel/d_target_pixel
    
    # considering misalignment
    #shift_x[i]=(1-scaling[i])*(4320/8)+((sigma*i*del_d_pixel*f_camera_pixel)/(d_list[i]))/4
    #shift_y[i]=(1-scaling[i])*(2880/8)+((gamma*i*del_d_pixel*f_camera_pixel)/(d_list[i]))/4
   
    shift_x[i]=(1-scaling[i])*(1080/2)
    shift_y[i]=(1-scaling[i])*(720/2)
    
    h=np.array([[scaling[i],0,shift_x[i]],[0,scaling[i],shift_y[i]],[0,0,1]])
    H[i]=(np.linalg.inv(h))

H_file = 'C:/Users/u6265553/Downloads/programs_codes/Hs.npz'
homographies = []
np.savez(H_file, homographies=H)
    
