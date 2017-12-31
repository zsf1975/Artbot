import numpy as np
import cv2

# Own modules
from utils import ProgressReport, fill_image


def dots(source_img):
    step = 20
    max_radius = 12
    background_color = [255, 255, 255]
    circle_color = [0, 0, 0]

    size_x = source_img.shape[1]
    size_y = source_img.shape[0]

    target_img = np.full(source_img.shape, 0, dtype=np.uint8)
    fill_image(target_img, background_color)

    status_print = ProgressReport(size_y, 'dots effect')
    report_step = status_print.get_step()

    for y in range(0, size_y, step):
        for x in range(0, size_x, step):
            rgb_avg = int(np.mean(source_img[y:y+step, x:x+step, 0:2]))
            radius = int((( 255.0 - rgb_avg) / 255.0) * max_radius)
            xcoord = int((x + step/2.0))
            ycoord = int((y + step/2.0))
            cv2.circle(target_img, (xcoord, ycoord), radius,\
                circle_color, thickness = -1, lineType=cv2.LINE_AA)

        status_print.update(y)

    status_print.finished()
    return target_img