import numpy as np
from random import randrange
import math
import cv2
from utils import ProgressReport, fill_image


def color_scribble(source_img):
    # This algorithm is mostly the same as mono_scribble. Mainly the color processing
    # is different. See the mono_scribble for more comprehensive comments. 


    # Background scaling coefficient. Value 1.0 will produce background color that is 
    # avarage of the entire original image. Values > 1 will brighten it. 
    # Values 0 - 1 will be darker that the original image. Value 0 will be black. 
    background_coeff = 0.0


    # Make working grayscale copy of the source img, since the working 
    # image will be destroyed during the process
    pic = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)

    # Image size params
    size_x = pic.shape[1]
    size_y = pic.shape[0]
    
    # Invert the working image and attenuate the high values to avoid 
    # white areas to be left untouched by the algorithm 
    # (all value 255 areas will be avoided by it).
    for y in range(size_y):
        for x in range(size_x):
            pix = 255 - pic[y, x]
            if(pix > 250):
                pix = 250   
            pic[y, x] = pix

    # Create empty array for the target image    
    background_color = np.mean(source_img, axis=(0,1)) * background_coeff
    img = np.full((size_y, size_x, 3), background_color, dtype=np.uint8)


    # Calculate estimate of the required line segment qty
    pic_mean = np.mean(pic)
    pix_qty = pic.shape[0] * pic.shape[1]
    loop_qty = int(pix_qty * (255 - pic_mean) / 2200)

    # Prepare the progress printing
    status_print = ProgressReport(loop_qty, 'color scribble effect')

    # Create look-up tables for sin and cos functions to speed-up the
    # processing.     
    sin360_table = []
    cos360_table = []
    for a in range(360):
        sin360_table.append(math.sin((a/180) * math.pi))
        cos360_table.append(math.cos((a/180) * math.pi))

    # Select random location for starting the drawing
    cur_x = randrange(size_x)
    cur_y = randrange(size_y)

    # Some working parameters
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
        rotat_step = 6 # degrees step for searching
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

                # Pick the pixel color from the original image
                if(0 < tx < size_x):
                    if(0 < ty < size_y):
                        pix = pic[ty, tx]

                # Check that is the pixel value the darkest found so far
                if(pix <= darkest_pix):
                    darkest_pix = pix
                    new_angle = tmp_a 
                    ptx = tx 
                    pty = ty 

            if(darkest_pix < 255): # True = something was found
                searching = False

            else: # Nothing usable was found, so expand the search  
                if(rotat_step > 1):
                    rotat_step -=1
                search_dist += 2
                range_max += 1 
            if(range_max >360): #360
                range_max = 360

            # If the search goes beyond the image area, it is likely
            # that all drawing is done and it is time to stop early. 
            if(search_dist > max(size_x, size_y)):
                early_stop = True
                searching = False
        # End of search loop.

        if(early_stop):
            break

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


        speed = 1.5
        nsteps = 10
        loc_bias = 0
        if(out_of_area):
            loc_bias = 0

        distance = speed * nsteps * 1.7                 
        delta_angle = new_angle - old_angle
        angle_step = delta_angle / nsteps
        prev_x = cur_x
        prev_y = cur_y
        old_x = cur_x 
        old_y = cur_y

        line_color = [0,0,0] # initial value just in case that the color picking fails. 
  
        for pt in range(nsteps):
            old_angle += angle_step
            angle = int(old_angle)

            cur_x += cos360_table[angle%360] * speed + loc_bias
            cur_y += sin360_table[angle%360] * speed + loc_bias
            
            # Pick the line segment color from original image.
            if((0 <= cur_x < size_x) and (0 <= cur_y <size_y)):
                line_color = source_img[int(cur_y), int(cur_x)].tolist()

            cv2.line(img, (int(cur_x), int(cur_y)),\
                (int(prev_x), int(prev_y)), line_color, 1, lineType=cv2.LINE_AA)

            prev_x = cur_x
            prev_y = cur_y
    
        if(t%100 == 0):
            status_print.update(t)

    status_print.finished()

    if(early_stop):
        print("Got the image processing ready faster than planned")

    print("Did draw total {} curves".format(loop_counter))

    return img 