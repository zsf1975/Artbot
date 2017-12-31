import numpy as np
import cv2
import math
from random import randrange
import sys

from utils import ProgressReport, fill_image, measure_time


def circles(source_img):

    # Some drawing variables
    packing = 1.05 # default = 1.05
    smooth_circles = True

    size_x = source_img.shape[1]
    size_y = source_img.shape[0]
    circle_list = [] 

    # Calculate background color for output image and initialize
    # the output image. 
    background_color = np.mean(source_img, axis=(0,1)) * 0.2
    target_img = np.full(source_img.shape, 0, dtype=np.uint8)
    fill_image(target_img, background_color)

    # Define one circle as a starting points.
    # All circles will inherit radius from this. 
    radius = 18
    active = True
    start_x = randrange(int(size_x / 2)) + int(size_x / 4)
    start_y = randrange(int(size_y / 2)) + int(size_y / 4)
    tstamp = 0
    circle_list.append([start_x, start_y, radius, active, tstamp])

    # Estimate the number of circles to be drawn. The 1350 is valid for circles 
    # with radius 18. 
    estimate = size_x * size_y / 1350
    status_print = ProgressReport(estimate, 'circles effect')

    # Make look-up tables to speed up the computing. 
    sin_list = []
    cos_list = []
    for a in range(360):
        sin_list.append(math.sin((a/180.0) * math.pi) * 2 * packing * radius)
        cos_list.append(math.cos((a/180.0) * math.pi) * 2 * packing * radius)

    search_range = 2 * radius
    circle_count = 0
    i = 0

    # Main loop for the algorithm
    while(i < len(circle_list)):

        if(i%10 == 0):
            # Report status
            status_print.update(circle_count)

        # Params for the active circle. New circles will born around this. 
        cpx = circle_list[i][0]
        cpy = circle_list[i][1]
        radius = circle_list[i][2]
        active = circle_list[i][3]

        # Process only active circles.
        if(active): 

            # Sweeo all angles around the circle to find places for new circles. 
            rand = randrange(360) 
            for angle in range(rand, 360+rand, 5):

                target_x = cpx + cos_list[angle%360]
                target_y = cpy + sin_list[angle%360]

                if((0 <= target_x < size_x)):
                    if((0 <= target_y < size_y)): # Split to two phases to improve speed
                        target_x = int(target_x)
                        target_y = int(target_y)

                        # Check that how much there is space available
                        avail_r, err = find_max_r(circle_list, (target_x, target_y),\
                            search_range, radius)
                        if(err == True):
                            continue
                        if(avail_r >= radius * packing):
                            # Create new circle
                            circle_count += 1 
                            circle_list.append([target_x, target_y, radius, True, i])      

                            if(smooth_circles):
                                l1_color = (source_img[target_y, target_x] * 1.0).tolist()
                                l2_color = (source_img[target_y, target_x] * 0.95).tolist()
                                l3_color = (source_img[target_y, target_x] * 0.90).tolist()
                                l4_color = (source_img[target_y, target_x] * 0.85).tolist()
                                l5_color = (source_img[target_y, target_x] * 0.75).tolist()
                                line_color = (source_img[target_y, target_x] * 0.5).tolist()

                                cv2.circle(target_img, (target_x, target_y), radius,\
                                    l5_color, thickness = -1, lineType=cv2.LINE_AA)

                                cv2.circle(target_img, (target_x, target_y), int(radius * 0.85),\
                                    l4_color, thickness = -1, lineType=cv2.LINE_AA)

                                cv2.circle(target_img, (target_x, target_y), int(radius * 0.7),\
                                    l3_color, thickness = -1, lineType=cv2.LINE_AA)

                                cv2.circle(target_img, (target_x, target_y), int(radius * 0.6),\
                                    l2_color, thickness = -1, lineType=cv2.LINE_AA)
                                cv2.circle(target_img, (target_x, target_y), int(radius * 0.4),\
                                    l1_color, thickness = -1, lineType=cv2.LINE_AA)

                                cv2.circle(target_img, (target_x, target_y), radius,\
                                    line_color, thickness = 1, lineType=cv2.LINE_AA)
                            else:
                                # Flat circle. 
                                l1_color = (source_img[target_y, target_x]).tolist()
                                line_color = (source_img[target_y, target_x] * 0.5).tolist()
                                cv2.circle(target_img, (target_x, target_y), radius,\
                                    l1_color, thickness = -1, lineType=cv2.LINE_AA)
                                cv2.circle(target_img, (target_x, target_y), radius,\
                                    line_color, thickness = 1, lineType=cv2.LINE_AA)
          
            # Deactivate the circle when all possible new circles have been generated around it
            circle_list[i][3] = False 


        # Remove very old circles from the cirle_list. This improves processing speed 10x - 100x when working 
        # with large images. The algorithm is not scientifically proven... It just happens to work. 
        # Target is to remove circles that are surrounded with circles, i.e. there is no point to keep those
        # included in the new circle calculations. 
        if(i%20 == 0):
            j = 0
            while(j < len(circle_list)):
                if(circle_list[j][4] < (i - 100) and circle_list[j][3] == False):
                    circle_list.pop(j)
                    i -= 1
                else:
                    j += 1
        
        i += 1
    
    status_print.finished()
    return target_img



def find_max_r(circle_list, coord, max_search, max_r):
    # This method scans through the circle list and finds the maximum available radius
    # in the new circle location (coord)

    # Center point for the studied circle location
    cp_x = coord[0]
    cp_y = coord[1]

    
    # Find the smalles distance from cp_x/y to any existing circle edge
    smallest_dist = None
    err = False
    for circle in circle_list:
        # x and y location and radius of the circle in the circle_list.
        x = circle[0]
        y = circle[1]
        r = circle[2]

        # Distance in x-direction
        dx = cp_x - x 
        
        # Break the search if the list items are too far in either direction
        if(dx > max_search):
            continue
        if(dx < -1 * max_search):
            continue

        # Distance in y-direction
        dy = cp_y - y 

        if(dy > max_search):
            continue
        if(dy < -1 * max_search):
            continue

        # Available distance
        dist = math.sqrt(dx**2 + dy**2) - r

        if(dist <= 0):
            # the location is inside other circle. 
            err = True
            break

        if(smallest_dist == None): 
            smallest_dist = dist

        if(dist <= smallest_dist):
            smallest_dist = dist 

    if(smallest_dist == None):
        # Nothin was found, i.e. there was no other circles around
        return max_r, err
    else: 
        # Return the available radius
        return int(smallest_dist), err


