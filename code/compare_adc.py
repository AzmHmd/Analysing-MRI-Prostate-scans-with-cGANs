import matplotlib.pyplot as plt 
import pydicom
import numpy as np 
import cv2 
from skimage.segmentation import clear_border 
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.color import label2rgb 
from functions_compare import *

fichier = open("adc/compare_final/results.txt","w")
fichier2 = open("adc/compare_final/results2.txt","w")
fichier.write("Case;Number;Dice noisy;Dice raw;Dice denoised;Dice raw+noisy;Dice raw+den;Dice raw+den+noisy;Haus noisy;Haus raw;Haus den;Haus raw+noisy;Haus raw+den;Haus raw+den+noisy\n")

fichier2.write("Case;Number;Noisy TP; FP;FN;TN;Raw TP;FP;FN;TN;Denoised TP;FP;FN;TN;Raw+noisy TP;FP;FN;TN;Raw+den TP;FP;FN;TN;Raw+den+noisy TP;FP;FN;TN\n")


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

TP_noisy = 0 
TP_raw = 0 
TP_denoised = 0
TP_raw_noisy = 0
TP_raw_denoised = 0 
TP_raw_denoised_noisy = 0 

FN_noisy = 0 
FN_raw = 0 
FN_denoised = 0
FN_raw_noisy = 0
FN_raw_denoised = 0 
FN_raw_denoised_noisy = 0 

TN_noisy = 0 
TN_raw = 0 
TN_denoised = 0
TN_raw_noisy = 0
TN_raw_denoised = 0 
TN_raw_denoised_noisy = 0 

for a in range (16,26) : 
    
    if a == 16 : 
        x1 = 24
        x2 = 16
        m = [2,16]
    if a == 17 : 
        x1 = 25
        x2 = 16
        m = [5,14]
    if a == 18 : 
        x1 = 26
        x2 = 16
        m = [8,16]
    if a == 19 : 
        x1 = 27
        x2 = 32
        m = [14,24]
    if a == 20 : 
        x1 = 29
        x2 = 32
        m = [12,25]
    if a == 21 : 
        x1 = 3
        x2 = 16
        m = [6,13]
    if a == 22 : 
        x1 = 30
        x2 = 32
        m = [17,27]
    if a == 23 :   
        x1 = 31
        x2 = 32
        m = [15,26]
    if a == 24 :  
        x1 = 32
        x2 = 32
        m = [11,28]
    if a == 25 :   
        x1 = 33
        x2 = 16
        m = [5,14]
    for i in range (1,x2+1):
        im = 'ST0000'+str(a)+' ('+str(x1)+')_%d' %i

        

        path_raw = 'adc/prostate_adc_raw_lr_low/'+nb+'_net_G_val/images/input/r'+im+'.png'
        path_denoised = 'adc/denoised/val/d'+im+'.png'
        path_noisy = 'adc/noisy/val/n'+im+'.png'
        path_mask_ori = 'adc/prostate_adc_raw_lr_low/'+nb+'_net_G_val/images/target/r'+im+'.png'
        path_mask_raw = 'adc/prostate_adc_raw_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_denoised = 'adc/prostate_adc_denoised_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_raw_denoised = 'adc/prostate_adc_raw_denoised_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_raw_denoised_noisy = 'adc/prostate_adc_raw_denoised_noisy_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_raw_noisy = 'adc/prostate_adc_raw_noisy_2_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png'
        path_mask_noisy = 'adc/prostate_adc_noisy_lr_low/'+nb+'_net_G_val/images/post/r'+im+'.png' 

        raw = cv2.imread(path_raw)
        denoised1 = cv2.imread(path_denoised)
        denoised = denoised1[:,0:256]
        noisy1 = cv2.imread(path_noisy)
        noisy=noisy1[:,0:256]
        mask_ori = cv2.imread(path_mask_ori)
        mask_raw = cv2.imread(path_mask_raw)
        mask_denoised =cv2.imread(path_mask_denoised)
        mask_raw_denoised =cv2.imread(path_mask_raw_denoised)
        mask_raw_denoised_noisy =cv2.imread(path_mask_raw_denoised_noisy)
        mask_raw_noisy =cv2.imread(path_mask_raw_noisy)
        mask_noisy =cv2.imread(path_mask_noisy)

        mask_ori_gray = cv2.imread(path_mask_ori,0)
        mask_raw_gray = cv2.imread(path_mask_raw,0)
        mask_denoised_gray =cv2.imread(path_mask_denoised,0)
        mask_raw_denoised_gray =cv2.imread(path_mask_raw_denoised,0)
        mask_raw_denoised_noisy_gray =cv2.imread(path_mask_raw_denoised_noisy,0)
        mask_raw_noisy_gray =cv2.imread(path_mask_raw_noisy,0)
        mask_noisy_gray =cv2.imread(path_mask_noisy,0)

        ##### Dice distance ---------------------------------------------------------
        mask_ori_bin =gray2binary (mask_ori_gray)
        mask_raw_bin =gray2binary (mask_raw_gray)
        mask_denoised_bin =gray2binary (mask_denoised_gray)
        mask_raw_denoised_bin =gray2binary (mask_raw_denoised_gray)
        mask_raw_denoised_noisy_bin =gray2binary (mask_raw_denoised_noisy_gray)
        mask_raw_noisy_bin =gray2binary (mask_raw_noisy_gray)
        mask_noisy_bin =gray2binary (mask_noisy_gray)

        ori_dice= np.reshape(mask_ori_bin,(-1))
        raw_dice = np.reshape(mask_raw_bin,(-1))
        denoised_dice = np.reshape(mask_denoised_bin,(-1))
        raw_denoised_dice = np.reshape(mask_raw_denoised_bin,(-1))
        raw_denoised_noisy_dice = np.reshape(mask_raw_denoised_noisy_bin,(-1))
        raw_noisy_dice = np.reshape(mask_raw_noisy_bin,(-1))
        noisy_dice = np.reshape(mask_noisy_bin,(-1))

        dist_dice_raw_c = distance.dice (ori_dice,raw_dice)
        dist_dice_denoised_c = distance.dice (ori_dice,denoised_dice) 
        dist_dice_raw_denoised_c = distance.dice (ori_dice,raw_denoised_dice) 
        dist_dice_raw_denoised_noisy_c = distance.dice (ori_dice,raw_denoised_noisy_dice) 
        dist_dice_raw_noisy_c = distance.dice (ori_dice,raw_noisy_dice) 
        dist_dice_noisy_c = distance.dice (ori_dice,noisy_dice)

        dist_dice_raw = (str(distance.dice (ori_dice,raw_dice)))[:6]
        dist_dice_denoised = (str(distance.dice (ori_dice,denoised_dice) ))[:6]
        dist_dice_raw_denoised = (str(distance.dice (ori_dice,raw_denoised_dice) ))[:6]
        dist_dice_raw_denoised_noisy = (str(distance.dice (ori_dice,raw_denoised_noisy_dice)))[:6] 
        dist_dice_raw_noisy = (str(distance.dice (ori_dice,raw_noisy_dice) ))[:6]
        dist_dice_noisy = (str(distance.dice (ori_dice,noisy_dice)  ))[:6]


        ### end Dice distance ---------------------------------------------

        ### Contours ---------------------------------------------------------

        img1,contours_ori,hierarchy1=cv2.findContours(mask_ori_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img2,contours_raw,hierarchy2=cv2.findContours(mask_raw_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img3,contours_denoised,hierarchy3=cv2.findContours(mask_denoised_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img4,contours_raw_denoised,hierarchy4=cv2.findContours(mask_raw_denoised_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img5,contours_raw_denoised_noisy,hierarchy5=cv2.findContours(mask_raw_denoised_noisy_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img6,contours_raw_noisy,hierarchy6=cv2.findContours(mask_raw_noisy_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        img7,contours_noisy,hierarchy7=cv2.findContours(mask_noisy_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        ### end Contours ----------------------------

        ## max Hausdorff distance -------------
        hausdorff_max_noisy = (str(redifined_distance_hausdorff (contours_ori, contours_noisy, maximum=True)))[:6]
        hausdorff_max_raw = (str(redifined_distance_hausdorff (contours_ori, contours_raw, maximum=True)))[:6]
        hausdorff_max_denoised = (str(redifined_distance_hausdorff (contours_ori, contours_denoised, maximum=True)))[:6]
        hausdorff_max_raw_noisy = (str(redifined_distance_hausdorff (contours_ori, contours_raw_noisy, maximum=True)))[:6]
        hausdorff_max_raw_denoised = (str(redifined_distance_hausdorff (contours_ori, contours_raw_denoised, maximum=True)))[:6]
        hausdorff_max_raw_denoised_noisy = (str(redifined_distance_hausdorff (contours_ori, contours_raw_denoised_noisy, maximum=True)))[:6]
        ## end max hausdorff distance ----------------------

        
        ## image building --------------------------

        add = np.zeros((256*7,256*6,3),dtype = np.uint8)

        #1st line 
        add[0:256,0:256] = raw
        add[0:256,256:256*2] = mask_ori
        add[0:256,256*2:256*3] = raw
        add[0:256,256*3:256*4] = raw
        add[0:256,256*4:256*5] = denoised
        add[0:256,256*5:256*6] = noisy

        redifined_print_contours(add, contours_ori, addx = 256*2, addy = 0, color2 = red , thickness2 = 1, lineType2 = 8)

        cv2.putText(add,'raw image',(10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'original mask',(256+10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'original contour',(256*2+10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'raw image',(256*3+10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'denoised image',(256*4+10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'noisy image',(256*5+10,20), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        #2nd line : trained on noisy images 
        add[256:256*2,256:256*2] = mask_noisy
        add[256:256*2,256*2:256*3] = raw
        add[256:256*2,256*3:256*4] = raw
        add[256:256*2,256*4:256*5] = raw

        redifined_print_contours(add, contours_noisy, addx = 256*2, addy = 256, color2 = red , thickness2 = 1, lineType2 = 8)
        final_hausdorff (add, contours_ori, contours_noisy, addy2=256)

        cv2.putText(add,'trained on noisy',(10,20+256), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'predicted mask',(256+10,20+256), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Dice distance :',(10+256*5,256+80), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_dice_noisy,(10+256*5,256+100), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Hausdorff distance max :',(10+256*5,256+140), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,hausdorff_max_noisy,(10+256*5,256+160), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        #3rd line : trained on raw images 
        add[256*2:256*3,256:256*2] = mask_raw
        add[256*2:256*3,256*2:256*3] = raw
        add[256*2:256*3,256*3:256*4] = raw
        add[256*2:256*3,256*4:256*5] = raw

        redifined_print_contours(add, contours_raw, addx = 256*2, addy = 256*2, color2 = red , thickness2 = 1, lineType2 = 8)
        final_hausdorff (add, contours_ori, contours_raw, addy2=256*2)

        cv2.putText(add,'trained on raw',(10,20+256*2), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Dice distance :',(10+256*5,256*2+80), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_dice_raw,(10+256*5,256*2+100), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Hausdorff distance max :',(10+256*5,256*2+140), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,hausdorff_max_raw,(10+256*5,256*2+160), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        #4th line : trained on denoised images 
        add[256*3:256*4,256:256*2] = mask_denoised
        add[256*3:256*4,256*2:256*3] = raw
        add[256*3:256*4,256*3:256*4] = raw
        add[256*3:256*4,256*4:256*5] = raw

        redifined_print_contours(add, contours_denoised, addx = 256*2, addy = 256*3, color2 = red , thickness2 = 1, lineType2 = 8)
        final_hausdorff (add, contours_ori, contours_denoised, addy2=256*3)

    
        cv2.putText(add,'trained on',(10,20+256*3), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'denoised',(10,40+256*3), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Dice distance :',(10+256*5,256*3+80), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_dice_denoised,(10+256*5,256*3+100), font, 0.5,(255,255,255),1,cv2.LINE_AA)


        cv2.putText(add,'Hausdorff distance max :',(10+256*5,256*3+140), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,hausdorff_max_denoised,(10+256*5,256*3+160), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        #5th line : trained on raw+noisy images 
        add[256*4:256*5,256:256*2] = mask_raw_noisy
        add[256*4:256*5,256*2:256*3] = raw
        add[256*4:256*5,256*3:256*4] = raw
        add[256*4:256*5,256*4:256*5] = raw

        redifined_print_contours(add, contours_raw_noisy, addx = 256*2, addy = 256*4, color2 = red , thickness2 = 1, lineType2 = 8)
        final_hausdorff (add, contours_ori, contours_raw_noisy, addy2=256*4)

        cv2.putText(add,'trained on raw ',(10,20+256*4), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'+ noisy',(10,40+256*4), font, 0.5,(255,255,255),1,cv2.LINE_AA)


        cv2.putText(add,'Dice distance :',(10+256*5,256*4+80), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_dice_raw_noisy,(10+256*5,256*4+100), font, 0.5,(255,255,255),1,cv2.LINE_AA)


        cv2.putText(add,'Hausdorff distance max :',(10+256*5,256*4+140), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,hausdorff_max_raw_noisy,(10+256*5,256*4+160), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        #6th line : trained on raw+denoised images 
        add[256*5:256*6,256:256*2] = mask_raw_denoised
        add[256*5:256*6,256*2:256*3] = raw
        add[256*5:256*6,256*3:256*4] = raw
        add[256*5:256*6,256*4:256*5] = raw

        redifined_print_contours(add, contours_raw_denoised, addx = 256*2, addy = 256*5, color2 = red , thickness2 = 1, lineType2 = 8)
        final_hausdorff (add, contours_ori, contours_raw_denoised, addy2=256*5)

        cv2.putText(add,'trained on ',(10,20+256*5), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'raw + denoised',(10,40+256*5), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Dice distance :',(10+256*5,256*5+80), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_dice_raw_denoised,(10+256*5,256*5+100), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Hausdorff distance max :',(10+256*5,256*5+140), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,hausdorff_max_raw_denoised,(10+256*5,256*5+160), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        #7th line : trained on raw+denoised+noisy images 
        add[256*6:256*7,256:256*2] = mask_raw_denoised_noisy
        add[256*6:256*7,256*2:256*3] = raw
        add[256*6:256*7,256*3:256*4] = raw
        add[256*6:256*7,256*4:256*5] = raw

        redifined_print_contours(add, contours_raw_denoised_noisy, addx = 256*2, addy = 256*6, color2 = red , thickness2 = 1, lineType2 = 8)
        final_hausdorff (add, contours_ori, contours_raw_denoised_noisy, addy2=256*6)

        cv2.putText(add,'trained on raw ',(10,20+256*6), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,'+ denoised + noisy',(10,40+256*6), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        
        cv2.putText(add,'Dice distance :',(10+256*5,256*6+80), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_dice_raw_denoised_noisy,(10+256*5,256*6+100), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.putText(add,'Hausdorff distance max :',(10+256*5,256*6+140), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,hausdorff_max_raw_denoised_noisy,(10+256*5,256*6+160), font, 0.5,(255,255,255),1,cv2.LINE_AA)

        cv2.imwrite('adc/compare_final/'+im+'.png',add)

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

            if dist_dice_raw_c != 1 :
                TN_raw += 1
                print('raw')
                print(a)
                print(i)
            if dist_dice_denoised_c != 1 :
                TN_denoised +=1
                print('denoised')
                print(a)
                print(i)
            if dist_dice_raw_denoised_c != 1 :
                TN_raw_denoised+=1
                print('raw denoised')
                print(a)
                print(i)
            if dist_dice_raw_denoised_noisy_c != 1 :
                TN_raw_denoised_noisy+=1
                print('raw denoised noisy')
                print(a)
                print(i)
            if dist_dice_raw_noisy_c != 1:
                TN_raw_noisy+=1
                print('raw noisy')
                print(a)
                print(i)
            if dist_dice_noisy_c != 1:
                TN_noisy+=1
                print('noisy')
                print(a)
                print(i)

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
        fichier2.write("\n")

        ##--------------------------------------------------

        
        if i >= m[0] and i <= m[1] :
           
            fichier.write(str(a)+';')
            fichier.write(str(i)+';')
            fichier.write(dist_dice_noisy+';')
            fichier.write(dist_dice_raw+';')
            fichier.write(dist_dice_denoised+';')
            fichier.write(dist_dice_raw_noisy+';')
            fichier.write(dist_dice_raw_denoised+';')
            fichier.write(dist_dice_raw_denoised_noisy+';')
            fichier.write(hausdorff_max_noisy+';')
            fichier.write(hausdorff_max_raw+';')
            fichier.write(hausdorff_max_denoised+';')
            fichier.write(hausdorff_max_raw_noisy+';')
            fichier.write(hausdorff_max_raw_denoised+';')
            fichier.write(hausdorff_max_raw_denoised_noisy+';')
            fichier.write("\n")


fichier.write("end \n")
fichier.close()

fichier2.write("end \n")
fichier2.close()








