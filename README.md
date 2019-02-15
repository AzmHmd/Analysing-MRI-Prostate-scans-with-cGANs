# Analysing-MRI-Prostate-scans-with-cGANs


Prostate cancer is the second most commonly diagnosed cancer among men and currently multi-parametric MRI is a promising imaging technique used for  clinical workup of prostate cancer. Accurate detection and localisation of  the prostate tissue boundary on various MRI scans can be helpful for obtaining a region of interest for Computer Aided Diagnosis systems.
In this paper, we present a fully automated detection and segmentation pipeline using a conditional Generative Adversarial Network (cGAN). We investigated the robustness of the cGAN model against adding Gaussian noise or removing  noise from the training data. Based on the  detection and segmentation metrics,  de-noising did not show a significant improvement. However, by including noisy images in the training data, the detection and segmentation performance was improved in each 3D modality, which resulted in  comparable to  state-of-the-art results. 


In the codes folder there are several functions for various tasks as bellow:

    binary.py 
        functions used to compute dice coefficient, hausdorff distance, jaccard index 

    functions_compare.py
        general functions used to analyse data :
        createFolder 
        transform an image in gray level to binary level 
        my personal function for hausdorff distance 
        print contours 
        print hausdorff distance 
        for post processing : keep giggest label 

main programs 

    postprocessing 
        do the post processing step and put images in the folder 'post' ; erase everything near the border and keep the biggest label
        postprocessing_adc.py
        postprocessing_dwi.py
        postprocessing_t2w.py

    compare 
        compare results from raw/noisy/denoised/raw+noisy/raw+denoised/raw+noisy+denoised 
        in image : big image with mask predicted, hausdorff distance printed, dice coeff        
        compare_adc.py
        compare_dwi.py
        compare_t2w.py
    
    det_metrics 
        TP,FP,TN,FN for all the datasets in the file results.txt
        det_metrics_adc.py
        det_metrics_dwi.py
        det_metrics_t2w.py

    seg_metrics 
        dice, jaccard, haus, haus 95 for each dataset im different files 
        seg_metrics_adc.py
        seg_metrics_dwi.py
        seg_metrics_t2w.py
