import numpy as np
import cv2
import math


def soften(source_img):
    kernel_size = 10
    sigma = 2
    filt_kernel = calculate_gaussian_kernel(kernel_size, kernel_size, sigma)
    result = cv2.filter2D(source_img, -1, filt_kernel)
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
    result = cv2.filter2D(source_img, -1, f_kernel)
    return result    


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
