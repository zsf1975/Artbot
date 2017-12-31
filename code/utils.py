import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import math
import sys
from random import randrange


def load_image(filename):
    # Loads image file and returns it in RGB format numpy array.
    img = cv2.imread(filename)
    if(len(img.shape) > 2 and img.shape[2] == 4):
        # Get rid of the aplha channel in PNG files. 
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        # Convert to RGB 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('Source image {} loaded'.format(filename))
    return img

def save_image(filename, img):
    # Save img numpy array to image file. 
    print('Saving the image to file {}'.format(filename))
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def add_dither_grayscale(img, amount):
    # Add dither to image pixels. Some effect algorithms may need this
    # if the image contains large areas of same color. 
    # The dither is ~zero mean in evenly distributed range +/- amount/2.
    # Currently works only with grayscale images. 

    size_x = img.shape[1]
    size_y = img.shape[0]
    result = img.copy()

    status_print = ProgressReport(size_y, 'dithering')

    for y in range(size_y):
        for x in range(size_x):
            pixel = int(img[y, x] + randrange(amount) - amount/2)
            if(pixel > 255): 
                pixel = 255
            elif(pixel < 0): 
                pixel = 0
            result[y, x] = pixel

        if(y%50 == 0):
            status_print.update(y)

    status_print.finished()
    return result


def plot_image(img):
    # Plot the image. This assumes that the image comes as numpy
    # array in RGB format. 
    print('Displaying image')
    plt.figure(figsize=(9,6))
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def fill_image(img, color):
    # Fill entire image with same color. 
    img[:, :] = color


def paint_area(img, y1, y2, x1, x2, color):
    # Paint rectangular area of the image. 
    img[y1:y2, x1:x2] = color


def histogram_equalize(source_img):
    # Takes grayscale image as input and equalizes the histogram.  
    
    print('Equalizing the histogram')
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    result = clahe.apply(source_img)
    return result


# Use this to measure execution time of some method. Just place the
# @measure_time decoration above the method definition. 
def measure_time(method):    
    def measure(*args, **kw):
        start_time = time.time()
        result = method(*args, **kw)
        end_time = time.time()
        print("{} {:.3f} s".\
            format(method.__name__, end_time - start_time))
        return result
    return measure


class ProgressReport(object):
    # This is used to report the processing progress to the user. 

    def __init__(self, finish_value, name):
        self.finish_value = finish_value
        sys.stdout.write('Computing ' + name + ':    ')
        sys.stdout.flush()
    
    def update(self, status_value):
        percent = status_value / self.finish_value * 100.0
        if(percent > 99.0):
            percent = 99.0
        sys.stdout.write('\b\b\b{:2.0f}%'.format(percent))
        sys.stdout.flush()

    def finished(self):
        sys.stdout.write('\b\b\bDone \n')
        sys.stdout.flush()        
