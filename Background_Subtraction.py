# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:57:51 2020

@author: Arif
"""

import cv2
from matplotlib import pylab as plt
import numpy as np
import glob
from PIL import Image

#import stack images without backlight
directory='F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/insect4_stack_img_feature_point_moving_lens_without_backlight_blender/*.jpg'
imglist=glob.glob(directory)
imglist.sort(reverse=False)

#import stack images with backlight
directory='F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/insect4_stack_img_feature_point_moving_lens_with_backlight_blender/*.jpg'
imglist_backlight=glob.glob(directory)
imglist_backlight.sort()

for i in range(len(imglist)):
    
    orig_img=cv2.resize(cv2.imread(imglist_backlight[i]),(1080,720))
    
    #img=orig_img[60:660,80:1000]
    img=orig_img
    hsv_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #plt.figure(1)
    #plt.imshow(hsv_img)
    
    # Generating mask
    ret,thresh1 = (cv2.threshold(hsv_img,200,255,cv2.THRESH_BINARY_INV))
    thresh=np.zeros([720,1080])
    #thresh[60:660,80:1000]=thresh1
    thresh=thresh1
    #thresh=np.float64(thresh1)
    #plt.figure(2)
    #plt.imshow(thresh)
    
    # save output mask
    if i<10:
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_moving_lens_feature_point_mask/fusion_pyramid'+str(0)+str(0)+str(i)+'_mask.png', cv2.resize(np.uint8(thresh),(1080*4,720*4)))
    elif all ([i>=10, i<100]):
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_moving_lens_feature_point_mask/fusion_pyramid'+str(0)+str(i)+'_mask.png', cv2.resize(np.uint8(thresh),(1080*4,720*4)))
    elif i>=100:
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_moving_lens_feature_point_mask/fusion_pyramid'+str(i)+'_mask.png', cv2.resize(np.uint8(thresh),(1080*4,720*4)))
    
    img1=cv2.resize(cv2.imread(imglist[i]),(1080,720))
    img2=np.ones([720,1080,3])
    img2[:,:,0]=img1[:,:,0]*(thresh/255)
    img2[:,:,1]=img1[:,:,1]*(thresh/255)
    img2[:,:,2]=img1[:,:,2]*(thresh/255)
    
    # saving stack images without backlight (necessary if the original image is cropped)
    if i<10:
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_without_backlight_moving_lens_feature_point_final/fusion_pyramid'+str(0)+str(0)+str(i)+'.jpg', cv2.resize(np.uint8(img1),(1080*4,720*4)))
    elif all ([i>=10, i<100]):
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_without_backlight_moving_lens_feature_point_final/fusion_pyramid'+str(0)+str(i)+'.jpg', cv2.resize(np.uint8(img1),(1080*4,720*4)))
    elif i>=100:
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_without_backlight_moving_lens_feature_point_final/fusion_pyramid'+str(i)+'.jpg', cv2.resize(np.uint8(img1),(1080*4,720*4)))
    
    #plt.figure(3)
    #plt.imshow(np.uint8(img2))
    
    ret,thresh=(cv2.threshold(thresh,100,255,cv2.THRESH_BINARY_INV))
    
    #plt.figure(4)
    #plt.imshow(thresh)
    
    img2[:,:,0]=img2[:,:,0]+thresh/1
    img2[:,:,1]=img2[:,:,1]+thresh
    img2[:,:,2]=img2[:,:,2]+thresh
    
    #img2=img2[40:680,60:1020]
    #img2=img2[60:660,80:1000]
    
    #plt.figure(5)
    #plt.imshow(np.uint8(img2))
        
    # saving background subtracted stack images
    if i<10:
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_moving_lens_feature_point_background_subtracted_blender/fusion_pyramid'+str(0)+str(0)+str(i)+'.jpg', cv2.resize(np.uint8(img2),(1080*4,720*4)))
    elif all ([i>=10, i<100]):
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_moving_lens_feature_point_background_subtracted_blender/fusion_pyramid'+str(0)+str(i)+'.jpg', cv2.resize(np.uint8(img2),(1080*4,720*4)))
    elif i>=100:
        cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/stack_img_moving_lens_feature_point_background_subtracted_blender/fusion_pyramid'+str(i)+'.jpg', cv2.resize(np.uint8(img2),(1080*4,720*4)))
    