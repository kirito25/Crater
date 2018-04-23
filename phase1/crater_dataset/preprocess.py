#!/usr/bin/python
"""
This will look at the images folder and the specified
technique to normalize the image and put the image in the normalize_image 
directory.
"""

import cv2 as cv
import numpy as np
import os, sys
import argparse
# Only import for debugging
#from matplotlib import pyplot as plt

BLUE = [255,0,0]
TARGET_SHAPE = (50, 50)
parser = argparse.ArgumentParser(description="Preprocess the images.")

parser.add_argument("-S", metavar='sizes', nargs="+", type=int, default=[50,50],
                            help="the size of the output image, default is [50, 50]")
parser.add_argument("-resize", dest="resize", action="store_true",
                            help="use the resize method, this is the default")
parser.add_argument("-border", dest="border", 
                            action="store_true", help="use the border method")
args = parser.parse_args()
if len(args.S) == 2:
    TARGET_SHAPE = tuple(args.S)

resize = True
if border:
    resize = False


SRC = "./crater_data/images/tile3_24/"
DST = "./crater_data/images/normalize_images/"
if not os.path.exists("./crater_data"):
    print "Please unzip the zip file"
    sys.exit()

i = 0
for directory in os.listdir(SRC):
    directory += "/"
    try:
        os.makedirs(DST + directory)
    except OSError:
        # directory already exist so do nothing
        pass
    for filename in os.listdir(SRC + directory):
        img1 = cv.imread(SRC + directory + filename)
        img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        top = bottom = (TARGET_SHAPE[0] - img1.shape[0] ) / 2
        right = left = (TARGET_SHAPE[1] - img1.shape[1] ) / 2
        constant = None
        if resize:
            constant = cv.resize(img1, TARGET_SHAPE)
            cv.normalize(constant, constant, 0, 255, cv.NORM_MINMAX)
        elif border:
            constant = cv.copyMakeBorder(img1, top, bottom, left, right, cv.BORDER_CONSTANT,value=BLUE)

        cv.imwrite(DST + directory + filename, constant)
        # For debuging purposes
        """
        print DST + directory + filename
        cv.imshow("dst_rt", constant)
        cv.waitKey(0)
        cv.destroyAllWindows()
        i += 1
        if (i > 10):
            sys.exit()
        """
