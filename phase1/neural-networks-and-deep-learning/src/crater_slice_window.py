from skimage.transform import pyramid_gaussian
import argparse
import cv2
import sys, time
#import imutils

image_1 = cv2.imread("../../crater_dataset/crater_data/images/tile3_24.pgm", 0)
image_2 = cv2.imread("../../crater_dataset/crater_data/images/tile3_25.pgm", 0)
(winW, winH) = (200, 200)


images = [image_1, image_2]


def test():
    for image in images:
        for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
        # if the image is too small, break from the loop
            if resized.shape[0] < 30 or resized.shape[1] < 30:
                break
                                    
            # show the resized image
            cv2.imshow("Layer {}".format(i + 1), resized)
            cv2.waitKey(0)


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for image in images:
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])



for image in images:
    for resized in pyramid_gaussian(image, downscale=2):
        if resized.shape[0] < winW or resized.shape[1] < winH:
            break
        for (x, y, window) in sliding_window(resized, stepSize=200, windowSize=(winW, winH)):

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
                    # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
                            # WINDOW
            
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            







