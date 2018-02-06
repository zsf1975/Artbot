
import numpy as np
import cv2
from scipy.spatial import Delaunay
from random import randrange

from utils import histogram_equalize, ProgressReport


def render_triangles(points, canvas, ref_img, BW):

    # Define color for triangle outlines.     
    if BW: 
        linecolor = 0
    else:
        linecolor = (0, 0, 0)

    x_max = canvas.shape[1]
    y_max = canvas.shape[0]

    # Calculate the triangles from the point cloud using Delaunay algorithm. 
    point_list = np.array(points)
    tri = Delaunay(point_list)
    tri = tri.simplices.copy()

    # Draw the filled triangles
    status_print = ProgressReport(len(tri), 'triangle fill rendering')
    i = 0
    for triangle in tri: 

        p0 = (point_list[triangle[0], 0], point_list[triangle[0], 1])
        p1 = (point_list[triangle[1], 0], point_list[triangle[1], 1])
        p2 = (point_list[triangle[2], 0], point_list[triangle[2], 1])
        cp = [int((p0[0] + p1[0] + p2[0]) / 3), int((p0[1] + p1[1] + p2[1]) / 3)]
        
        if BW:
            color = int(ref_img[cp[1], cp[0]])
        else:
            color = (int(ref_img[cp[1], cp[0], 0]),\
                     int(ref_img[cp[1], cp[0], 1]),\
                     int(ref_img[cp[1], cp[0], 2]))

        cv2.fillConvexPoly(canvas, np.array([p0, p1, p2]), color=color)
        
        if i%1000 == 0:
            status_print.update(i)
        i += 1
    status_print.finished()

    # Draw the triangle outlines. This is done separately 
    # in order to avoid the fill painting over outlines. 
    status_print = ProgressReport(len(tri), 'triangle outline rendering')
    i = 0
    for triangle in tri: 

        p0 = (point_list[triangle[0], 0], point_list[triangle[0], 1])
        p1 = (point_list[triangle[1], 0], point_list[triangle[1], 1])
        p2 = (point_list[triangle[2], 0], point_list[triangle[2], 1])

        cv2.line(canvas, p0, p1, linecolor, thickness = 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p1, p2, linecolor, thickness = 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, p2, p0, linecolor, thickness = 1, lineType=cv2.LINE_AA)
 
        if i%1000 == 0:
            status_print.update(i)
        i += 1
    status_print.finished()


    print("Drawing the image using {} triangles".format(len(tri)))

    # Return the image in RGB format.
    if BW: 
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    else:
        return canvas

def triangulate(source_img, BW):
    # Calculate triangulated effect. Both colored and grayscale versions use this same code. 
    # BW = True for grayscale processing.

    # Calculate temporary image that is used to calculate triangle points. 
    tmp_img = cv2.cvtColor(source_img, cv2.COLOR_RGB2GRAY)
    tmp_img = histogram_equalize(tmp_img)

    # Define the color reference images and canvas for the final image. 
    # Canvas is painted to the average gray value of the source image. 
    if BW: 
        ref_img = tmp_img.copy()
        mean_color = np.mean(ref_img)
        canvas = np.full(ref_img.shape, mean_color, dtype=np.uint8)
    else:
        ref_img = source_img.copy()
        mean_color = np.mean(ref_img)
        canvas = np.full(ref_img.shape, (mean_color, mean_color, mean_color), dtype=np.uint8)

    y_max = tmp_img.shape[0]
    x_max = tmp_img.shape[1]
    point_list = []
    tmp_img = tmp_img * 0.98
    tmp_img = tmp_img + 2
    rand_points = 1000000

    status_print = ProgressReport(rand_points, 'triangle corner points')

    # Define the point cloud that is used to draw the triangles. 
    # Million randomly located points are used in the process. Most of those are ignored. 
    for pt in range(rand_points):

        if pt%1000 == 0:
            status_print.update(pt)

        y = randrange(y_max)
        x = randrange(x_max) 

        # Pic color from temp image.
        color = tmp_img[y, x]

        # Calculate upper limit for the color processnig. 
        # At the beginning of the processing only dark values are taken into account. 
        # Later the limit is increased towards light colors and finally all colors are analyzed. 
        limit = pt / 3000 + 1
        if color > limit:
            continue

        # Ignore colors that are below 2. Those are considered as already processed and this algorith
        # does process same location only once. 
        if color < 2:
            continue

        # Calculate radius for the are that will be marked as processed to the temp image and 
        # draw the area to the temp image. 
        radius = int(color/6) + 5
        cv2.circle(tmp_img, (x+2, y+2), radius, 0, thickness = -1)

        # Add the poit location to point list. 
        point_list.append([x, y])

    status_print.finished()

    # Render the triangles
    result = render_triangles(point_list, canvas, ref_img, BW)
    return result


