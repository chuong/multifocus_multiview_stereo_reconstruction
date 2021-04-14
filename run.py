#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 19:35:56 2019

@author: chuong nguyen <chuong.nguyen@csiro.au>
"""

'''It performs the image stacking task. It first align the images given the 
transformaton function. We used ECC algorithm to align the images. Then it 
blended the aligned images using Laplacian pyramid fusion algorithm to obtain 
fully in-focus images.'''

import sys, os, glob
import alignments
import fusions
import pyramid
import cv2
import numpy as np

if __name__ == '__main__':
    folders=13
    #subfolders=24
    
    for i in range(folders):
        subfolders=int(len(glob.glob('F:/Arif/moving_lens/blender_images/insect4/Moving_lens_with_backlight_64/img/x='+str(i)+'/*/*.jpg'))/64) #61)
        for j in range(subfolders):
            
            folder='F:/Arif/moving_lens/blender_images/insect4/Moving_lens_with_backlight_64/img/x='+str(i)+'/y='+str(j)
            extensions = ['*.png', '*.jpg']
            files = []
            [files.extend(glob.glob(os.path.join(folder, ext))) for ext in extensions]
            #files.sort(reverse=True)
            files.sort(reverse=False)
            print('Found:\n' + '\n'.join(files))
            image_BGRs=[]
            image_BGRs = [cv2.resize(cv2.imread(file),(1080,720)) for file in files]
                    
            # load precomputed homographies if available
            H_file = 'F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/Hs.npz'
                        
            homographies = []
            if os.path.isfile(H_file):
                homographies = np.load(H_file)['homographies']
        
            # compute homographies, if not available, and align images
            aligned_BGRs=[]
            #aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'ECC', homographies)
            #aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'ECC_PYRAMID')
            aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'SIFT', homographies)
            #aligned_BGRs, homographies = alignments.align_images(image_BGRs, 'HYBRID')
        
            # save homographies if not yet done
            if not os.path.isfile(H_file):
                np.savez(H_file, homographies=homographies)
                
            # Fuse the aligned images
            #F = fusions.fuse_simple(aligned_BGRs)
            #F = fusions.fuse_guided_filter(aligned_BGRs)
            F = pyramid.get_pyramid_fusion(np.asarray(aligned_BGRs))
        
            #save the stacked images
            if all([j<10,i<10]):
                cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/insect4_stack_img_feature_point_moving_lens_with_backlight_blender/fusion_pyramid'+str(0)+str(i)+str(0)+str(j)+'.jpg', cv2.resize(F,(1080*4,720*4)))
            elif all([j>=10,i<10]):
                cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/insect4_stack_img_feature_point_moving_lens_with_backlight_blender/fusion_pyramid'+str(0)+str(i)+str(j)+'.jpg', cv2.resize(F,(1080*4,720*4)))
            elif all([j<10,i>=10]):
                cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/insect4_stack_img_feature_point_moving_lens_with_backlight_blender/fusion_pyramid'+str(i)+str(0)+str(j)+'.jpg', cv2.resize(F,(1080*4,720*4)))
            elif all([j>=10,i>=10]):
                cv2.imwrite('F:/Arif/moving_lens/blender_images/insect4/feature_point_approach/insect4_stack_img_feature_point_moving_lens_with_backlight_blender/fusion_pyramid'+str(i)+str(j)+'.jpg', cv2.resize(F,(1080*4,720*4)))
            
            print('Done!')
