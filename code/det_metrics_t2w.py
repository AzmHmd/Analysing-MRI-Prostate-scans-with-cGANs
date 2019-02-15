import matplotlib.pyplot as plt 
import pydicom
import numpy as np 
import cv2 
from skimage.segmentation import clear_border 
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.color import label2rgb 
from functions_compare import *

fichier2 = open("t2w/results.txt","w")

fichier2.write("Case;Number;Noisy TP; FP;FN;TN;Raw TP;FP;FN;TN;Denoised TP;FP;FN;TN;Raw+noisy TP;FP;FN;TN;Raw+den TP;FP;FN;TN;Raw+den+noisy TP;FP;FN;TN;Noisy2 TP; FP;FN;TN;raw+Noisy2 TP; FP;FN;TN;Noisy3 TP; FP;FN;TN;Raw+Noisy3 TP; FP;FN;TN;denoised2 TP; FP;FN;TN;raw+denoised2 TP; FP;FN;TN;\n")

nb = str(50)
red = (0,0,255)
blue = (255,0,0)
green = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX

FP_noisy = 0 
FP_raw = 0 
FP_denoised = 0
FP_raw_noisy = 0
FP_raw_denoised = 0 
FP_raw_denoised_noisy = 0 
FP_noisy2 = 0 
FP_raw_noisy2 = 0 
FP_noisy3 = 0 
FP_raw_noisy3 = 0 
FP_denoised2 = 0 
FP_raw_denoised2 = 0 

TP_noisy = 0 
TP_raw = 0 
TP_denoised = 0
TP_raw_noisy = 0
TP_raw_denoised = 0 
TP_raw_denoised_noisy = 0 
TP_noisy2 = 0 
TP_raw_noisy2 = 0 
TP_noisy3 = 0 
TP_raw_noisy3 = 0 
TP_denoised2 = 0 
TP_raw_denoised2 = 0 

FN_noisy = 0 
FN_raw = 0 
FN_denoised = 0
FN_raw_noisy = 0
FN_raw_denoised = 0 
FN_raw_denoised_noisy = 0 
FN_noisy2 = 0 
FN_raw_noisy2 = 0 
FN_noisy3 = 0 
FN_raw_noisy3 = 0 
FN_denoised2 = 0 
FN_raw_denoised2 = 0 


TN_noisy = 0 
TN_raw = 0 
TN_denoised = 0
TN_raw_noisy = 0
TN_raw_denoised = 0 
TN_raw_denoised_noisy = 0 
TN_noisy2 = 0 
TN_raw_noisy2 = 0 
TN_noisy3 = 0 
TN_raw_noisy3 = 0 
TN_denoised2 = 0 
TN_raw_denoised2 = 0 

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
        im = 'ST0000'+str(a)+' ('+str(x1)+')_%d' %i

        path_raw = 't2w/prostate_t2w_raw_lr_low/'+nb+'_net_G_val/images/input/r'+im+'.png'
        path_denoised = 't2w/denoised/val/d'+im+'.png'
        path_noisy = 't2w/noisy/val/n'+im+'.png'
        path_mask_ori = 't2w/prostate_t2w_raw_lr_low/'+nb+'_net_G_val/images/target/r'+im+'.png'
        path_mask_raw = 't2w/prostate_t2w_raw_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_denoised = 't2w/prostate_t2w_denoised_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_raw_denoised = 't2w/prostate_t2w_raw_denoised_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_raw_denoised_noisy = 't2w/prostate_t2w_raw_denoised_noisy_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_raw_noisy = 't2w/prostate_t2w_raw_noisy_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_noisy = 't2w/prostate_t2w_noisy_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 
        path_mask_noisy2 = 't2w/prostate_t2w_noisy2_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 
        path_mask_raw_noisy2 = 't2w/prostate_t2w_raw_noisy2_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 
        path_mask_noisy3 = 't2w/prostate_t2w_noisy3_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 
        path_mask_raw_noisy3 = 't2w/prostate_t2w_raw_noisy3_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 
        path_mask_denoised2 = 't2w/prostate_t2w_denoised2_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 
        path_mask_raw_denoised2 = 't2w/prostate_t2w_raw_denoised2_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 

        mask_ori_gray = cv2.imread(path_mask_ori,0)
        mask_raw_gray = cv2.imread(path_mask_raw,0)
        mask_denoised_gray =cv2.imread(path_mask_denoised,0)
        mask_raw_denoised_gray =cv2.imread(path_mask_raw_denoised,0)
        mask_raw_denoised_noisy_gray =cv2.imread(path_mask_raw_denoised_noisy,0)
        mask_raw_noisy_gray =cv2.imread(path_mask_raw_noisy,0)
        mask_noisy_gray =cv2.imread(path_mask_noisy,0)
        mask_noisy2_gray =cv2.imread(path_mask_noisy2,0)
        mask_raw_noisy2_gray =cv2.imread(path_mask_raw_noisy2,0)
        mask_noisy3_gray =cv2.imread(path_mask_noisy3,0)
        mask_raw_noisy3_gray =cv2.imread(path_mask_raw_noisy3,0)
        mask_denoised2_gray =cv2.imread(path_mask_denoised2,0)
        mask_raw_denoised2_gray =cv2.imread(path_mask_raw_denoised2,0)

        mask_ori_bin =gray2binary (mask_ori_gray)
        mask_raw_bin =gray2binary (mask_raw_gray)
        mask_denoised_bin =gray2binary (mask_denoised_gray)
        mask_raw_denoised_bin =gray2binary (mask_raw_denoised_gray)
        mask_raw_denoised_noisy_bin =gray2binary (mask_raw_denoised_noisy_gray)
        mask_raw_noisy_bin =gray2binary (mask_raw_noisy_gray)
        mask_noisy_bin =gray2binary (mask_noisy_gray)
        mask_noisy2_bin =gray2binary (mask_noisy2_gray)
        mask_raw_noisy2_bin =gray2binary (mask_raw_noisy2_gray)
        mask_noisy3_bin =gray2binary (mask_noisy3_gray)
        mask_raw_noisy3_bin =gray2binary (mask_raw_noisy3_gray)
        mask_denoised2_bin =gray2binary (mask_denoised2_gray)
        mask_raw_denoised2_bin =gray2binary (mask_raw_denoised2_gray)

        ori_dice= np.reshape(mask_ori_bin,(-1))
        raw_dice = np.reshape(mask_raw_bin,(-1))
        denoised_dice = np.reshape(mask_denoised_bin,(-1))
        raw_denoised_dice = np.reshape(mask_raw_denoised_bin,(-1))
        raw_denoised_noisy_dice = np.reshape(mask_raw_denoised_noisy_bin,(-1))
        raw_noisy_dice = np.reshape(mask_raw_noisy_bin,(-1))
        noisy_dice = np.reshape(mask_noisy_bin,(-1))
        noisy2_dice = np.reshape(mask_noisy2_bin,(-1))
        raw_noisy2_dice = np.reshape(mask_raw_noisy2_bin,(-1))
        noisy3_dice = np.reshape(mask_noisy3_bin,(-1))
        raw_noisy3_dice = np.reshape(mask_raw_noisy3_bin,(-1))
        denoised2_dice = np.reshape(mask_denoised2_bin,(-1))
        raw_denoised2_dice = np.reshape(mask_raw_denoised2_bin,(-1))


        dist_dice_raw_c = distance.dice (ori_dice,raw_dice)
        dist_dice_denoised_c = distance.dice (ori_dice,denoised_dice) 
        dist_dice_raw_denoised_c = distance.dice (ori_dice,raw_denoised_dice) 
        dist_dice_raw_denoised_noisy_c = distance.dice (ori_dice,raw_denoised_noisy_dice) 
        dist_dice_raw_noisy_c = distance.dice (ori_dice,raw_noisy_dice) 
        dist_dice_noisy_c = distance.dice (ori_dice,noisy_dice)
        dist_dice_noisy2_c = distance.dice (ori_dice,noisy2_dice)
        dist_dice_raw_noisy2_c = distance.dice (ori_dice,raw_noisy2_dice)
        dist_dice_noisy3_c = distance.dice (ori_dice,noisy3_dice)
        dist_dice_raw_noisy3_c = distance.dice (ori_dice,raw_noisy3_dice)
        dist_dice_denoised2_c = distance.dice (ori_dice,denoised2_dice)
        dist_dice_raw_denoised2_c = distance.dice (ori_dice,raw_denoised2_dice)

        img1,contours_ori,hierarchy1=cv2.findContours(mask_ori_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



        ##### TP,FP,TN,FN -------------------------------------------
 


        #FP and TN 

        if (len(contours_ori)== 0 ) : 

            if dist_dice_raw_c == 1 :
                FP_raw += 1
            if dist_dice_denoised_c == 1 :
                FP_denoised +=1
            if dist_dice_raw_denoised_c == 1 :
                FP_raw_denoised+=1
            if dist_dice_raw_denoised_noisy_c == 1 :
                FP_raw_denoised_noisy+=1
            if dist_dice_raw_noisy_c == 1 :
                FP_raw_noisy+=1
            if dist_dice_noisy_c == 1:
                FP_noisy+=1
            if dist_dice_noisy2_c == 1:
                FP_noisy2+=1
            if dist_dice_raw_noisy2_c == 1:
                FP_raw_noisy2+=1
            if dist_dice_noisy3_c == 1:
                FP_noisy3+=1
            if dist_dice_raw_noisy3_c == 1:
                FP_raw_noisy3+=1
            if dist_dice_denoised2_c == 1:
                FP_denoised2+=1
            if dist_dice_raw_denoised2_c == 1:
                FP_raw_denoised2+=1

            if dist_dice_raw_c != 1 :
                TN_raw += 1
            if dist_dice_denoised_c != 1 :
                TN_denoised +=1
            if dist_dice_raw_denoised_c != 1 :
                TN_raw_denoised+=1
            if dist_dice_raw_denoised_noisy_c != 1 :
                TN_raw_denoised_noisy+=1
            if dist_dice_raw_noisy_c != 1:
                TN_raw_noisy+=1
            if dist_dice_noisy_c != 1:
                TN_noisy+=1
            if dist_dice_noisy2_c != 1:
                TN_noisy2+=1
            if dist_dice_raw_noisy2_c != 1:
                TN_raw_noisy2+=1
            if dist_dice_noisy3_c != 1:
                TN_noisy3+=1
            if dist_dice_raw_noisy3_c != 1:
                TN_raw_noisy3+=1
            if dist_dice_denoised2_c != 1:
                TN_denoised2+=1
            if dist_dice_raw_denoised2_c != 1:
                TN_raw_denoised2+=1

        if (len(contours_ori)!= 0 ) : 
            if dist_dice_raw_c == 1 :
                FN_raw += 1
            if dist_dice_denoised_c == 1 :
                FN_denoised +=1
            if dist_dice_raw_denoised_c == 1 :
                FN_raw_denoised+=1
            if dist_dice_raw_denoised_noisy_c == 1 :
                FN_raw_denoised_noisy+=1
            if dist_dice_raw_noisy_c == 1 :
                FN_raw_noisy+=1
            if dist_dice_noisy_c == 1:
                FN_noisy+=1
            if dist_dice_noisy2_c == 1:
                FN_noisy2+=1
            if dist_dice_raw_noisy2_c == 1:
                FN_raw_noisy2+=1
            if dist_dice_noisy3_c == 1:
                FN_noisy3+=1
            if dist_dice_raw_noisy3_c == 1:
                FN_raw_noisy3+=1
            if dist_dice_denoised2_c == 1:
                FN_denoised2+=1
            if dist_dice_raw_denoised2_c == 1:
                FN_raw_denoised2+=1
            
            if dist_dice_raw_c != 1 :
                TP_raw += 1
            if dist_dice_denoised_c != 1 :
                TP_denoised +=1
            if dist_dice_raw_denoised_c != 1 :
                TP_raw_denoised+=1
            if dist_dice_raw_denoised_noisy_c != 1 :
                TP_raw_denoised_noisy+=1
            if dist_dice_raw_noisy_c != 1 :
                TP_raw_noisy+=1
            if dist_dice_noisy_c != 1:
                TP_noisy+=1
            if dist_dice_noisy2_c != 1:
                TP_noisy2+=1
            if dist_dice_raw_noisy2_c != 1:
                TP_raw_noisy2+=1
            if dist_dice_noisy3_c != 1:
                TP_noisy3+=1
            if dist_dice_raw_noisy3_c != 1:
                TP_raw_noisy3+=1
            if dist_dice_denoised2_c != 1:
                TP_denoised2+=1
            if dist_dice_raw_denoised2_c != 1:
                TP_raw_denoised2+=1


        fichier2.write(str(a)+';')
        fichier2.write(str(i)+';')
        fichier2.write(str(TP_noisy)+';')
        fichier2.write(str(FP_noisy)+';')
        fichier2.write(str(FN_noisy)+';')
        fichier2.write(str(TN_noisy)+';')
        fichier2.write(str(TP_raw)+';')
        fichier2.write(str(FP_raw)+';')
        fichier2.write(str(FN_raw)+';')
        fichier2.write(str(TN_raw)+';')
        fichier2.write(str(TP_denoised)+';')
        fichier2.write(str(FP_denoised)+';')
        fichier2.write(str(FN_denoised)+';')
        fichier2.write(str(TN_denoised)+';')
        fichier2.write(str(TP_raw_noisy)+';')
        fichier2.write(str(FP_raw_noisy)+';')
        fichier2.write(str(FN_raw_noisy)+';')
        fichier2.write(str(TN_raw_noisy)+';')
        fichier2.write(str(TP_raw_denoised)+';')
        fichier2.write(str(FP_raw_denoised)+';')
        fichier2.write(str(FN_raw_denoised)+';')
        fichier2.write(str(TN_raw_denoised)+';')
        fichier2.write(str(TP_raw_denoised_noisy)+';')
        fichier2.write(str(FP_raw_denoised_noisy)+';')
        fichier2.write(str(FN_raw_denoised_noisy)+';')
        fichier2.write(str(TN_raw_denoised_noisy)+';')

        fichier2.write(str(TP_noisy2)+';')
        fichier2.write(str(FP_noisy2)+';')
        fichier2.write(str(FN_noisy2)+';')
        fichier2.write(str(TN_noisy2)+';')

        fichier2.write(str(TP_raw_noisy2)+';')
        fichier2.write(str(FP_raw_noisy2)+';')
        fichier2.write(str(FN_raw_noisy2)+';')
        fichier2.write(str(TN_raw_noisy2)+';')

        fichier2.write(str(TP_noisy3)+';')
        fichier2.write(str(FP_noisy3)+';')
        fichier2.write(str(FN_noisy3)+';')
        fichier2.write(str(TN_noisy3)+';')

        fichier2.write(str(TP_raw_noisy3)+';')
        fichier2.write(str(FP_raw_noisy3)+';')
        fichier2.write(str(FN_raw_noisy3)+';')
        fichier2.write(str(TN_raw_noisy3)+';')

        fichier2.write(str(TP_denoised2)+';')
        fichier2.write(str(FP_denoised2)+';')
        fichier2.write(str(FN_denoised2)+';')
        fichier2.write(str(TN_denoised2)+';')

        fichier2.write(str(TP_raw_denoised2)+';')
        fichier2.write(str(FP_raw_denoised2)+';')
        fichier2.write(str(FN_raw_denoised2)+';')
        fichier2.write(str(TN_raw_denoised2)+';')
        fichier2.write("\n")

        ##--------------------------------------------------

        
fichier2.write("end \n")
fichier2.close()
