import matplotlib.pyplot as plt
import pydicom
import numpy as np
import cv2
import scipy.spatial.distance as distance 
from skimage.segmentation import mark_boundaries
from skimage.segmentation import clear_border 
from skimage.measure import label
import os 

def createFolder (directory) : 
    try : 
        if not os.path.exists(directory) :
            os.makedirs(directory)
    except OSError :
        print('Error : Creating directory.' + directory) 



red = (0,0,255)
blue = (255,0,0)
green = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
def gray2binary (image) : 

    ret,image_bin = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    for i in range(256):
        for j in range (256):
            if image_bin[i][j]==255:
                image_bin[i][j]=1
    return (image_bin)

def binary2gray (image) : 
    image_gray = np.zeros((256,256),dtype = np.uint8)
    for i in range(256):
        for j in range (256):
            if image[i][j]==1:
                image_gray[i][j]=255
    return (image_gray)


#### Hausdorff -----------------------------------------------------------------------------------------

def contours2coord (contours) :

    pts = np.zeros((len(contours[0]),2),dtype=int)
    for i in range (0, len(contours[0])):
      
        pts[i][0] = contours[0][i][0][0]
        pts[i][1] = contours[0][i][0][1]

    return(pts)


def redifined_distance_hausdorff (contours_ori, contours_pred, maximum=True, sens='ori2pred' ) : 
    if ((len(contours_pred)!= 0 ) and (len(contours_ori)!= 0 ))  :

        pts_ori = contours2coord(contours_ori)
        pts_pred = contours2coord(contours_pred)
           
        ori_pred = distance.directed_hausdorff(pts_ori,pts_pred)
        pred_ori = distance.directed_hausdorff(pts_pred,pts_ori)

        if maximum :
            return max (ori_pred[0], pred_ori[0])
           # if ori_pred[0] < pred_ori[0] :
            #    return (pred_ori)
           # else :
           #     return (ori_pred)

        else : 
            if sens == 'ori2pred' : 
                return (ori_pred)
            elif sens == 'pred2ori' :
                return (pred_ori)

def redifined_print_contours(image, contours, addx = 0, addy = 0, color2 = (0,0,255) , thickness2 = 1, lineType2 = 8) : 
    if len(contours)!=0 :
        x = []
        y = []
        for i in range (0,len(contours[0])):
            x.append(contours[0][i][0][0])
            y.append(contours[0][i][0][1])

        for i in range (0, len(x)-1):
            cv2.line(image,(x[i]+addx,y[i]+addy), (x[i+1]+addx,y[i+1]+addy), color2, thickness = thickness2, lineType = lineType2)
        cv2.line(image,(x[len(x)-1]+addx,y[len(x)-1]+addy), (x[0]+addx,y[0]+addy), color2, thickness = thickness2, lineType = lineType2)


def redifined_print_contours2(image, contours, addx = 0, addy = 0, color2 = (0,0,255) , thickness2 = 1, lineType2 = 8) : 
    if len(contours)!=0 :
        for k in range(0,len(contours)):
            x = []
            y = []
            for i in range (0,len(contours[k])):
                x.append(contours[k][i][0][0])
                y.append(contours[k][i][0][1])

            for i in range (0, len(x)-1):
                cv2.line(image,(x[i]+addx,y[i]+addy), (x[i+1]+addx,y[i+1]+addy), color2, thickness = thickness2, lineType = lineType2)
            cv2.line(image,(x[len(x)-1]+addx,y[len(x)-1]+addy), (x[0]+addx,y[0]+addy), color2, thickness = thickness2, lineType = lineType2)






def redifined_print_hausdorff (image, contours_ori, contours_pred, dist_hausdorff, addx = 0, addy = 0, color2 = (0,255,0) , thickness2 = 1, lineType2 = 8) : 

    pts_ori = contours2coord(contours_ori)
    pts_pred = contours2coord(contours_pred)

    indice_couple_ori = dist_hausdorff[1]
    couple_ori = pts_ori[indice_couple_ori]
    x_ori = couple_ori[0]
    y_ori = couple_ori[1]
    
    indice_couple_pred = dist_hausdorff[2]
    couple_pred = pts_pred[indice_couple_pred]
    x_pred = couple_pred[0]
    y_pred = couple_pred[1]
    
    coordPt1= (x_ori+addx,y_ori+addy)
    coordPt2 = (x_pred+addx,y_pred+addy)

    cv2.line(image,coordPt1,coordPt2,color2, thickness = thickness2, lineType = lineType2)

################-----------------------------------------------------------------------------------------

####### Post processing 

def keep_biggest_label (labels, num) :
    list_labels = []

    if num == 0 :  
        mask_return = np.zeros((256,256),dtype = np.uint8)
        return (mask_return)

    else : 
        for k in range(num):
            somme = 0
            for i in range(256):
                for j in range (256):     
                    if labels[i][j]==k:
                        somme += 1
            list_labels.append(somme)

        r = list_labels.index(max(list_labels))

        mask_return = np.zeros((256,256),dtype = np.uint8)
        for i in range(64,192):
            for j in range (64,192):
                if labels[i][j]==r+1:
                    mask_return[i][j]=1
                  
        return (mask_return)


####hausdorff distance : compare_final 

def final_hausdorff (add, contours_ori, contours_post, addy2, addx2=256*3) : 

    if ((len(contours_post)!= 0 ) and (len(contours_ori)!= 0 ))  : 
        dist_hausdorff = redifined_distance_hausdorff (contours_ori, contours_post, maximum=False,sens = 'ori2pred' )
        redifined_print_contours(add, contours_ori, addx =addx2 , addy = addy2,color2 = red , thickness2 = 1, lineType2 = 8) 
        redifined_print_contours(add, contours_post, addx = addx2, addy = addy2,color2 = blue , thickness2 = 1, lineType2 = 8) 
        redifined_print_hausdorff (add, contours_ori, contours_post, dist_hausdorff, addx = addx2, addy = addy2, color2 = green , thickness2 = 1, lineType2 = 8) 
        dist = dist_hausdorff[0]
        dist_str = str(dist) 
        dist_tronk = dist_str[:6]


        dist_hausdorff2 = redifined_distance_hausdorff (contours_ori, contours_post, maximum=False,sens = 'pred2ori' )
        redifined_print_contours(add, contours_ori, addx = addx2+256, addy = addy2,color2 = red , thickness2 = 1, lineType2 = 8) 
        redifined_print_contours(add, contours_post, addx = addx2+256, addy = addy2,color2 = blue , thickness2 = 1, lineType2 = 8) 
        redifined_print_hausdorff (add, contours_post, contours_ori, dist_hausdorff2, addx = addx2+256, addy =addy2, color2 = green , thickness2 = 1, lineType2 = 8)
        dist2 = dist_hausdorff2[0]
        dist_str2 = str(dist2) 
        dist_tronk2 = dist_str2[:6]


        cv2.putText(add,'Hausdorff distance :',(10+addx2,addy2+10), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_tronk,(addx2+10,addy2+20+10), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(add,dist_tronk2,(addx2+10+256,addy2+20+10), font, 0.5,(255,255,255),1,cv2.LINE_AA)














