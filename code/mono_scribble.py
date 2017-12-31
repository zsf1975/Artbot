import numpy as np
from random import randrange
import math
import cv2
from utils import ProgressReport


def mono_scribble(source_img):

    # This effect takes grayscale image as input. 

    # Define target color theme:
    # Sepia theme
    #line_color = [20, 10, 1]
    #background_color = [240, 240, 230]

    # Black & white theme
    line_color = [0, 0, 0]
    background_color = [255, 255, 255]

    # Make copy of the source img, since the working omage will be 
    # destroyed during the process
    pic = source_img.copy()

    # Max values for x and y. 
    size_x = pic.shape[1]
    size_y = pic.shape[0]
    
    # Attenuate the high values to avoid white areas to be left untouched.
    # The algorithm won't draw areas that are 255. 
    for y in range(size_y):
        for x in range(size_x):
            if(pic[y, x] > 250):
                pic[y, x] = 250   

    # Create empty array for the target image
    img = np.full((size_y, size_x, 3), background_color, dtype=np.uint8)
    
    # Calculate estimate of the required line segment qty
    # This is empirical formula and typically overestimates the complexity, 
    # which means that the algorithm stops by itself before reaching this limit. 
    pic_mean = np.mean(pic)
    pix_qty = pic.shape[0] * pic.shape[1]
    loop_qty = int(pix_qty * (255 - pic_mean) / 2200)

    # Prepare the progress printing
    status_print = ProgressReport(loop_qty, 'mono scribble effect')
    
    # Create look-up tables for sin and cos. This speeds those functions by 
    # approx 30%, which can be significant saving. 
    sin360_table = []
    cos360_table = []
    for a in range(360):
        sin360_table.append(math.sin((a/180) * math.pi))
        cos360_table.append(math.cos((a/180) * math.pi))


    # Select random location for starting the drawing
    cur_x = randrange(size_x)
    cur_y = randrange(size_y)

    # Some starting params. 
    new_angle = 0 
    old_angle = 0 
    speed = 0.4
    old_angle = 0
    new_angle = 0
    distance = speed * 50 
    clearing_radius = 4

    # Start the main drawing loop 
    early_stop = False
    loop_counter = 0
    for t in range(loop_qty):
        loop_counter += 1

        # Calculate new direction for the line by searching the 
        # darkest spot aroung the current location
        searching = True
        rotat_step = 6 # degree step size for searching
        search_dist = distance # how far to look in searching
        range_max = 180 # initial angle range for the search

        while(searching):
            darkest_pix = 255 # initialize the max possible value

            # This loop scans the original image for new direction
            for a in range(0, range_max, rotat_step): # a = angle in degrees

                tmp_a = int(old_angle + a - range_max/2) # temporary angle variable
                tx = int(cur_x + cos360_table[tmp_a%360] * search_dist)
                ty = int(cur_y + sin360_table[tmp_a%360] * search_dist)
                pix = 255

                # Pick the pixel color from the image
                if(0 < tx < size_x):
                    if(0 < ty < size_y):
                        pix = pic[ty, tx]

                # Check that is the pixel value the darkest found so far
                if(pix <= darkest_pix):
                    darkest_pix = pix
                    new_angle = tmp_a 
                    ptx = tx 
                    pty = ty 

            # Check that was there new direction found
            if(darkest_pix < 255): 
                searching = False # Stop the search.
            else: # Nothing usable was found, so expand the search range
                if(rotat_step > 1):
                    rotat_step -=1
                search_dist += 2
                range_max += 1 
            if(range_max >360):
                range_max = 360

            # If the search goes beyond the image area, it is likely
            # that all drawing is done and it is time to stop early. 
            # All pixels in the pic array should be 255 in this phase.
            # This is the primary way to stop the algorithm. 
            if(search_dist > max(size_x, size_y)):
                early_stop = True
                searching = False
        # End of search loop.

        if(early_stop):
            break   # Stop the drawing loop

        # Check that has the algorithm gone nuts and doing something 
        # outside the image area. Doesn't happen often... 
        # If it happens, the direction is forced towards the image center.
        out_of_area = False
        if((ptx < 0) or (ptx > size_x) or (pty < 0) or (pty > size_y)):
            dx = size_x / 2 - dir_x
            dy = size_y / 2 - dir_y
            new_angle = math.atan2(dy, dx)
            out_of_area = True


        # Clear the found aim point location in the original picture. This 
        # ensures that we won't be drawing the same area many times. 
        # Note that this may be bit different location than the line end point.
        clearing_color = darkest_pix + 80
        clearing_radius = int(clearing_color/30)
        if(clearing_color > 255):
            clearing_color = 255

        cv2.circle(pic, (ptx, pty), clearing_radius, int(clearing_color), -1)


        # Line drawing parameters
        speed = 1.5 # Drawing speed
        nsteps = 10 # Line segments per one curve
        loc_bias = 0  # Can use bias in the location calculation

        if(out_of_area):
            loc_bias = 0

        # Distance estimate for the search algorithm
        distance = speed * nsteps * 1.7 
                 
        delta_angle = new_angle - old_angle
        angle_step = delta_angle / nsteps
        prev_x = cur_x
        prev_y = cur_y
        old_x = cur_x 
        old_y = cur_y
    
        # Draw the curve to the line
        for pt in range(nsteps):
            old_angle += angle_step
            angle = int(old_angle)
            cur_x += cos360_table[angle%360] * speed + loc_bias
            cur_y += sin360_table[angle%360] * speed + loc_bias
            
            cv2.line(img, (int(cur_x), int(cur_y)),\
                (int(prev_x), int(prev_y)), line_color, 1, lineType=cv2.LINE_AA)

            prev_x = cur_x
            prev_y = cur_y
    
        # Update the progress report
        if(t%100 == 0):
            status_print.update(t)

    status_print.finished()

    if(early_stop):
        print("Got the image processing ready faster than planned")

    print("Did draw total {} curves".format(loop_counter))

    return img 
