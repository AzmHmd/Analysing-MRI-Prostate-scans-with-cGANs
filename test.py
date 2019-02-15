import numpy as np
import matplotlib.pyplot as plt
import pydicom
import cv2

def original(path) :
    
    ds = pydicom.dcmread(path)
    raw1 = ds.pixel_array
    raw2 = raw1 - raw1.min()
    coeff2 = 255./raw2.max()
    raw3 = raw2*coeff2
    raw4 = raw3.astype(np.uint8)
    
    return (raw4)


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

def cut (image):


#ori = original(path)
path ='adc/denoised/train/ST000000 (1)_8.png'
image = cv2.imread(path,0)
ori=image[:][0:256]
mask_prostate = image[:][256:512]
#ori = cv2.imread(path)
#path_mask = ''
#mask = cv2.imread(path_mask,0)
mask_prostate_bin = gray2binary(mask)
precise_prostate = (np.multiply(ori,mask_prostate_bin))

cv2.imwrite('',precise_prostate)