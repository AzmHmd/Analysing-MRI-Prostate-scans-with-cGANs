import matplotlib.pyplot as plt 
import pydicom
import numpy as np 
import cv2 
from skimage.segmentation import clear_border 
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.color import label2rgb 
from functions_compare import *

nb = str(50)
for k in range(8,12):
    if k ==0 : 
        fold = 'raw'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 1 : 
        fold = 'noisy'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 2 : 
        fold = 'denoised_2'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 5 : 
        fold = 'raw_denoised'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 4 : 
        fold = 'raw_noisy'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 3 : 
        fold = 'raw_denoised_noisy_2'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 6 : 
        fold = 'noisy2'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 7 : 
        fold = 'raw_noisy2'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 8 :
        fold = 'noisy3'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 9 : 
        fold = 'raw_noisy3'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k ==10 : 
        fold = 'denoised2'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)
    if k == 11 : 
        fold = 'raw_denoised2'
        createFolder ('./t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/')
        print(fold)


    for a in range (16,26) : 
        if a == 16 : 
            x1 = 24
            x2 = 30
        if a == 17 : 
            x1 = 25
            x2 = 30
        if a == 18 : 
            x1 = 26
            x2 = 30
        if a == 19 : 
            x1 = 27
            x2 = 30
        if a == 20 : 
            x1 = 29
            x2 = 30
        if a == 21 : 
            x1 = 3
            x2 = 28
        if a == 22 : 
            x1 = 30
            x2 = 34
        if a == 23 :   
            x1 = 31
            x2 = 32
        if a == 24 :  
            x1 = 32
            x2 = 32
        if a == 25 :   
            x1 = 33
            x2 = 22
        for i in range (1,x2+1):
            im = 'rST0000'+str(a)+' ('+str(x1)+')_%d' %i
            path_raw = 't2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/input/'+im+'.png'
            path_mask_ori = 't2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/target/'+im+'.png'
            path_mask_pred = 't2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/output/'+im+'.png'

            raw= cv2.imread(path_raw)
            mask_ori = cv2.imread(path_mask_ori)
            mask_pred = cv2.imread(path_mask_pred)

            mask_ori1 = cv2.imread(path_mask_ori,0)
            mask_pred1 = cv2.imread(path_mask_pred,0)

            mask_pred_bin = gray2binary (mask_pred1)
            cleared_bin = clear_border (mask_pred_bin)
            labels, num = label(cleared_bin, return_num = True)
            mask_return_bin= keep_biggest_label (labels, num) 
            mask_return_gray = binary2gray (mask_return_bin)
           
            cv2.imwrite('t2w/prostate_t2w_'+fold+'_lr_low/'+nb+'_net_G_val/images/post/'+im+'.png',mask_return_gray)
