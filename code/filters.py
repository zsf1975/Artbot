import numpy as np
import cv2
import math

from utils import ProgressReport

# This file contains sharpen and soften filter implementations. 
# Very slow processing, but allow you to tweak everything. 
# These work on RGB images. 


def soften(source_img):
    kernel_size = 10
    sigma = 2
    filt_kernel = calculate_gaussian_kernel(kernel_size, kernel_size, sigma)
    result = conv_filter(source_img, filt_kernel, 'soften') 
    return result


def sharpen(source_img):   
    f_kernel = np.array([\
        [-1, -1, -1, -1, -1],\
        [-1,  2,  2,  2, -1],\
        [-1,  2,  8,  2, -1],\
        [-1,  2,  2,  2, -1],\
        [-1, -1, -1, -1, -1]])
    kernel_sum = np.sum(f_kernel)
    scaling_coeff = 1.0 / kernel_sum
    f_kernel = f_kernel * scaling_coeff    
    result = conv_filter(source_img, f_kernel, 'sharpen') 
    return result    


def conv_filter(source_img, filter_kernel, description):
    y_max = source_img.shape[0]
    x_max = source_img.shape[1]

    kernel_size = filter_kernel.shape[0] 
    kernel_w1 = int(kernel_size/2)
    kernel_w2 = kernel_size - kernel_w1 +1
    target_img = np.full(source_img.shape, 0, dtype=np.uint8)
    temp_img = np.pad(source_img, ((kernel_w1, kernel_w2),\
        (kernel_w1, kernel_w2), (0, 0)), mode='edge') 

    status_print = ProgressReport(y_max + kernel_size, description)

    for y in range(0, y_max, 1):
        for x in range(0, x_max, 1):
            sr = np.sum(np.multiply(temp_img[y : y+kernel_size,\
                x : x+kernel_size, 0].astype(np.float32),\
                filter_kernel.astype(np.float32)))
            
            sg = np.sum(np.multiply(temp_img[y : y+kernel_size,\
                x : x+kernel_size, 1].astype(np.float32),\
                filter_kernel.astype(np.float32)))
            
            sb = np.sum(np.multiply(temp_img[y : y+kernel_size,\
                x : x+kernel_size, 2].astype(np.float32),\
                filter_kernel.astype(np.float32)))
            
            if(sr > 255):
                sr = 255
            elif(sr < 0):
                sr = 0
            if(sg > 255):
                sg = 255
            elif(sg < 0):
                sg = 0
            if(sb > 255):
                sb = 255
            elif(sb < 0):
                sb = 0
            target_img[y, x, 0] = sr 
            target_img[y, x, 1] = sg
            target_img[y, x, 2] = sb

        if(y%10 == 0):
            status_print.update(y)


    status_print.finished()
    return target_img


def calculate_gaussian_kernel(size_x, size_y, sigma):
    size_x = size_x
    size_y = size_y
    sigma = sigma
    fkernel = np.full((size_y, size_x), 0.0)
    cp_x = size_x / 2.0 
    cp_y = size_y / 2.0 
    p1 = 1.0 / (2 * math.pi * sigma * sigma) 
    pdiv = 2.0 * sigma * sigma

    for y in range(size_y):
        for x in range(size_x):
            ep = -1.0 * ((x - cp_x)**2.0 + (y - cp_y)**2.0) / pdiv
            value = p1 * math.e ** ep
            fkernel[y, x] = value

    # Scale the kernel so that all elements sum to 1.0 
    kernel_sum = np.sum(fkernel)
    scaling_coeff = 1.0 / kernel_sum
    fkernel = fkernel * scaling_coeff        
    return fkernel
