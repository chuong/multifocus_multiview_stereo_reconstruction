#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:45:01 2019

@author: chuong nguyen <chuong.nguyen@csiro.au>
"""
import cv2
#from cv2.ximgproc import guidedFilter
import numpy as np
from timed import timed
import os, glob


@timed
def filter_LoG(grays, gaussian_size=5, gaussian_sigma=0, laplacian_size=5):
    gaussians = [cv2.GaussianBlur(gray, (gaussian_size, gaussian_size), gaussian_sigma)
                 for gray in grays]
    LoGs = [cv2.Laplacian(gaussian, cv2.CV_64F, laplacian_size)
            for gaussian in gaussians]
#    laplacians = [cv2.Laplacian(gray, cv2.CV_64F, laplacian_size)
#                  for gray in grays]
#    LoGs = [cv2.GaussianBlur(laplacian, (gaussian_size, gaussian_size), gaussian_sigma)
#            for laplacian in laplacians]
    LoGs_array = np.asarray(LoGs)
    LoGs_array = LoGs_array + 1e-12
    LoGs_array = LoGs_array/LoGs_array.sum(axis=0)[None, :, :]  # normalise to 1
    return list(LoGs)


@timed
def filter_average(grays, kernel_size=5):
    return [cv2.blur(gray, (kernel_size, kernel_size)) for gray in grays]

'''
@timed
def filter_guided(weight_maps, grays, radius=45, eps=0.3):
    Ws = [guidedFilter(gray.astype(np.float32)/255, weight, radius, eps).clip(0, 1)
          for weight, gray in zip(weight_maps, grays)]
    Ws_array = np.asarray(Ws)
    Ws_array = Ws_array + 1e-12
    Ws_array = Ws_array/Ws_array.sum(axis=0)[None, :, :]  # normalise to 1
    return list(Ws_array)
'''

@timed
def filter_fft_cv(gray, mask):
    # apply mask and inverse DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)

    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    print(img_back.min(), img_back.max()) # strange output value
    return img_back.clip(0, 255).astype(gray.dtype)


@timed
def filter_fft_np(gray, mask):
    f = np.fft.fft2(gray)

    fshift = np.fft.fftshift(f)
    fshift = fshift*mask[:, :, 0]
    f_ishift = np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back.clip(0, 255).astype(gray.dtype)


@timed
def get_base_detail_fft(imgs, radius=15):
    '''Trial function to see if this produce better/fast base and detail layers
    It turns out that both speed and quality are worse than average filter
    '''
    rows, cols = imgs[0].shape[:2]

    # get optimised width and height
    nrow = cv2.getOptimalDFTSize(rows)
    ncol = cv2.getOptimalDFTSize(cols)

    # create mask
    crow, ccol = nrow//2, ncol//2
    mask = np.zeros((nrow, ncol, 2), np.float32)
#    mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
    cv2.circle(mask, (ccol, crow), radius, 1, cv2.FILLED)
    mask = cv2.blur(mask, (5, 5))
    cv2.imshow('mask', mask[:,:,0])

    details = []
    bases = []
    for img in imgs:
        if len(img.shape) == 2:
            nimg = np.zeros((nrow, ncol), img.dtype)
            nimg[:rows, :cols] = img
            bases.append(filter_fft_np(nimg, mask)[:rows, :cols])
            details.append(filter_fft_np(nimg, 1-mask)[:rows, :cols])
        else:
            nimg = np.zeros((nrow, ncol, img.shape[2]), img.dtype)
            nimg[:rows, :cols, :] = img
            detail = np.zeros_like(nimg)
            base = np.zeros_like(nimg)
            for i in range(img.shape[2]):
                base[:, :, i]   = filter_fft_np(nimg[:, :, i], mask)
                detail[:, :, i] = filter_fft_np(nimg[:, :, i], 1-mask)
            details.append(detail[:rows, :cols, :])
            bases.append(base[:rows, :cols, :])

    return details, bases


if __name__ == "__main__":
    folder = '../data/Demo_from_Helicon_Focus'
    extensions = ['*.png', '*.jpg']
    files = []
    [files.extend(glob.glob(os.path.join(folder, ext))) for ext in extensions]
    files.sort()
    print('Found:\n' + '\n'.join(files))
    image_BGRs = [cv2.imread(file) for file in files]
    details, bases = get_base_detail_fft(image_BGRs, radius=31)
    for detail, base in zip(details, bases):
        cv2.imshow('detail', detail)
        cv2.imshow('base', base)
        cv2.waitKey()
    cv2.destroyAllWindows()