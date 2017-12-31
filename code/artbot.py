# Standard modules 
import cv2
import sys
import os.path

from utils import load_image, save_image, plot_image
from utils import histogram_equalize, add_dither_grayscale
from filters import soften, sharpen
from dots import dots
from circles import circles
from mono_scribble import mono_scribble
from color_scribble import color_scribble


def main(argv):

    # Parse the command line arguments
    if(len(argv) != 3):
        print_help()
        sys.exit()
    source_filename = argv[0]
    destination_filename = argv[1]
    effect = argv[2]

    # Load the source image
    if(os.path.isfile(source_filename)):
        img = load_image(source_filename)
    else:
        print("Wrong input file name!")
        print_help()
        sys.exit()

    # Process the image
    if(effect == "circles"):
        print("Processing image")
        scaling = 3000 / max(img.shape[0], img.shape[1])
        tmp = cv2.resize(img, (0,0), fx=scaling, fy=scaling)        
        result = circles(tmp)

    elif(effect == "dots"):
        print("Processing image")
        scaling = 4000 / max(img.shape[0], img.shape[1])
        tmp = cv2.resize(img, (0,0), fx=scaling, fy=scaling)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
        tmp = histogram_equalize(tmp) # works only on grayscale images
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
        result = dots(tmp)

    elif(effect == "scribble"):
        print("Processing image")
        scaling = 3000 / max(img.shape[0], img.shape[1])
        tmp = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #tmp = add_dither_grayscale(tmp, 3) # may be needed for images with flat color areas
        tmp = cv2.resize(tmp, (0,0), fx=scaling, fy=scaling)
        tmp = histogram_equalize(tmp)
        result = mono_scribble(tmp)

    elif(effect == "color_scribble"):
        print("Processing image")
        scaling = 3000 / max(img.shape[0], img.shape[1])
        tmp = cv2.resize(img, (0,0), fx=scaling, fy=scaling)
        result = color_scribble(tmp)

    else: 
        print("Wrong effect name!")
        print_help()
        sys.exit()

    # Show the image and then save it
    plot_image(result)
    save_image(destination_filename, result)


def print_help(): 
    
    docstring = """
Run the program using command: 
python3 artbot.py inputfile outputfile effect
    
where: 
    inputfile is the input image file name. 
    outputfile is the filename that is used to save the result image. 
    effect is the name of the processing effect. 

File formats: 
    Supports all image formats (jpg, png, etc) that are supported by OpenCV. 
    
Effects: 
    circles
        Draws colored circles on dark background. 
    dots
        Produces black dots on white background. 
    scribble
        Draws single color scribble line on white background.
    color_scribble
        Draws colored line on black background. 
    """
    print(docstring)


if(__name__ == "__main__"):
    main(sys.argv[1:])
