from skimage.transform import pyramid_gaussian
import argparse
import cv2

image_1 = cv2.imread("../../crater_dataset/crater_data/images/tile3_24.pgm", 0)
image_2 = cv2.imread("../../crater_dataset/crater_data/images/tile3_25.pgm", 0)

images = [image_1] #, image_2]

for image in images:
    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=2)):
        # if the image is too small, break from the loop
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break
                                
        # show the resized image
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)

