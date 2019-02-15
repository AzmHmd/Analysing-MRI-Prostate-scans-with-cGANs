#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 12:33:34 2018

@author: Amelie
"""

import numpy as np
import cv2

def gray2binary (image) :
    ret,image_bin = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    for i in range(256):
        for j in range (256):
            if image_bin[i][j]==255:
                image_bin[i][j]=1
    return (image_bin)

def gray2binaryv2 (image) :
    
    ret,image_bin = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    for i in range(512):
        for j in range (512):
            if image_bin[i][j]==255:
                image_bin[i][j]=1
    return (image_bin)

path ='ST000000 (1)_7.png'
image1 = cv2.imread(path,0)
image2=cv2.imread(path)
ori=image2[:,0:256]
mask = image1[:,256:512]
mask_prostate_bin = gray2binary(mask)
_, contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#precise_prostate = (np.multiply(ori,mask_prostate_bin))
cv2.drawContours(ori, contour, -1, (0, 0, 255), 2)
cv2.imwrite('coucou.png',ori)