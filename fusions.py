#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:46:15 2019

@author: chuong nguyen <chuong.nguyen@csiro.au>
"""

import cv2
import numpy as np
import filters
from timed import timed


@timed
def fuse_simple(colors,
                radius_gaussian=2, sigma_gaussian=0, laplacian_size=3):
    '''
    Simple fusion using Laplacian of Gaussian
    '''
    # compute global weight maps
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in colors]
#    grays = [cv2.blur(gray, (3, 3)) for gray in grays]
    LoGs = filters.filter_LoG(grays, 2*radius_gaussian+1, 2*sigma_gaussian+1,
                              laplacian_size)
    LoGs_array = np.asarray(LoGs)
    ALoGs = np.absolute(LoGs_array)
    maximum = ALoGs.max(axis=0)
    P_array = (ALoGs == maximum).astype(np.uint8)
    P = list(P_array)

    output = np.zeros(shape=colors[0].shape, dtype=colors[0].dtype)
    for i in range(len(colors)):
        output = cv2.bitwise_not(colors[i], output, mask=P[i])
#        cv2.imshow('mask', 255*P[i])
#        cv2.waitKey()

    return 255-output


@timed
def fuse_guided_filter(colors, base_size=50,
                       radius_gaussian=2, sigma_gaussian=2, laplacian_size=3,
                       radii=[21, 21], epsilons=[0.3, 1e-6]):
    '''
    Ref:
    Li, S., Kang, X., & Hu, J. (2013).
    Image fusion with guided filtering.
    IEEE Transactions on Image processing, 22(7), 2864-2875.
    '''
    # compute global weight maps
    grays = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in colors]
    LoGs = filters.filter_LoG(grays, 2*radius_gaussian+1, sigma_gaussian,
                              laplacian_size)
    LoGs_array = np.asarray(LoGs)

    # make sure only one element per pixel on weight maps
    LoGs_array = np.absolute(LoGs_array)
    P_array = np.zeros_like(LoGs_array, np.uint8)
    ind = np.indices(LoGs_array.shape)
    P_array[LoGs_array.argmax(axis=0), ind[1], ind[2]] = 1
    P = list(P_array)

    # compute weight maps of base and detail layers
    W_bases   = filters.filter_guided(P, colors, radii[0], epsilons[0])
    W_details = filters.filter_guided(P, colors, radii[1], epsilons[1])
    bases = filters.filter_average(colors, base_size)
    details = [color.astype(np.float32) - base.astype(np.float32)
               for color, base in zip(colors, bases)]
#    details, bases = filters.get_base_detail_fft(colors, radius=50)

#    for i in range(len(bases)):
#        cv2.imshow('P_i', (255*P[i]).astype(np.uint8))
#        cv2.imshow('base_i', bases[i].astype(np.uint8))
#        cv2.imshow('detail_i', details[i].astype(np.uint8))
#        cv2.imshow('W_base_i', (255*W_bases[i]).astype(np.uint8))
#        cv2.imshow('W_detail_i', (255*W_details[i]).astype(np.uint8))
#        cv2.waitKey()

    # fusion as weight sum of base and detail layes
    base = [W_base[:, :, np.newaxis]*base
            for W_base, base in zip(W_bases, bases)]
    base = np.asarray(base).sum(axis=0)
    detail = [W_detail[:, :, np.newaxis]*detail
              for W_detail, detail in zip(W_details, details)]
    detail = np.asarray(detail).sum(axis=0)
    cv2.imshow('base', base.astype(np.uint8))
    cv2.imshow('detail', detail.astype(np.uint8))
    F = base + detail
#    F = F/F.max()*255
#    F = F.clip(0, 255)
#    F = (F - F.min())/(F.max()-F.min())*255
    F = F.clip(0, 255).astype(np.uint8)
    return F
