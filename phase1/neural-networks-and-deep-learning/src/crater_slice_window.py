import argparse
parser = argparse.ArgumentParser(description="Run object detection for craters experiment.")
parser.add_argument("-s", metavar='s', type=int, default=200 , help="size of the square sliding window, default is 200")
parser.add_argument("-stepsize", metavar='stepsize', type=int, default=10 , help="the step size for sliding window, default is 10")
parser.add_argument("-show", dest="show", action="store_true", help="use to show the moving window graphically, default is no")
args = parser.parse_args()

(winW, winH) = (int(args.s), int(args.s))
window_size = (winH, winW)
stepsize = int(args.stepsize)
show = args.show

import cv2
import sys, time
import keras_cnn
import numpy as np
from helpers import *

image_1 = cv2.imread("../../crater_dataset/crater_data/images/tile3_24.pgm")
image_2 = cv2.imread("../../crater_dataset/crater_data/images/tile3_25.pgm")
images = [image_1] #, image_2]

def crater_sliding_window(image, stepSize):
    layer = 0
    for image_layer in pyramid(image, scale=2):
        layer += 1
        for (x, y, window) in sliding_window(image_layer, stepSize, window_size):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            yield (x, y, window, image_layer, layer)
        cv2.imwrite("img_layer_%d.jpeg" % (layer), image_layer)

model = keras_cnn.network()
og_image = image_1.copy()
prev_layer = 0
for (x, y, window, image_layer, layer) in crater_sliding_window(image_1, stepsize):
    if prev_layer != layer:
        print "Working on layer %d" % (layer)
        prev_layer = layer
    if np.argmax(model.predict(keras_cnn.cvImg2input(window))) == 1:
        cv2.rectangle(image_layer,  (x, y), (x + winW, y + winH), (0, 255, 0), 2)

        cv2.rectangle(og_image,  (x * layer * layer, y * layer * layer), 
                                    ((x + winW) * layer * layer, (y + winH) * layer * layer), 
                                    (0, 255, 0), 2)
    if show:
        copy = image_layer.copy()
        cv2.rectangle(copy, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("window", copy)
        cv2.waitKey(1)
        time.sleep(0.025)

print "Writing resulting image to test.jpeg"
cv2.imwrite("test.jpeg", og_image)
