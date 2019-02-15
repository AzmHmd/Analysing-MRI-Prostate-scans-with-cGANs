
functions 

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
