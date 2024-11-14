# Methods in this module are used to segment a coin image into individual coins.

import matplotlib.pyplot as plt
import cv2
import numpy as np

############################## Bilinear interpolation #########################
# This method receives values at the 4 corners of a rectangle: 
# (r1,c1,x11), (r1,c2,x12), (r2,c1,x21), (r2,c2,x22)
# and determines the value x at location (r, c) within the rectangle.
# by bilinear interpolation.
def bilinear(r, c, r1,r2,c1, c2, x11,x12,x21,x22):
    x = (x11 * (r2 - r) * (c2 - c) +
            x21 * (r - r1) * (c2 - c) +
            x12 * (r2 - r) * (c - c1) +
            x22 * (r - r1) * (c - c1)
           ) / ((r2 - r1) * (c2 - c1) + 0.0)
    return x

####################### Segment image by bilinear Otsu thresholding ############
# This method implements the adaptive image segmentation method of Otsu.
# An image is divided into 4 quadrants, the threshold value for segmentation
# of each quadrant is determined by the method of Otsu. The threshold values
# are then used to estimate threshold value at each pixel within the
# image by bilinear interpolation/extrapolation of thresholds obtained within the 
# 4 quadrants.
def thresholdingBilinearOtsu(img):
    
    # Partition img into 4 equal quadrants
    nr,nc = img.shape
    nr2 = int(nr/2)
    nc2 = int(nc/2)
    quad = img[0:nr2, 0:nc2]                # quadrant 11
    thd11,thresholded = cv2.threshold(quad, # Image to be segmented
                               0,           # Initial dark region intensity
                               255,         # Initial bright region intensity
                               cv2.THRESH_OTSU)      # Create a binary image with 0 showing intensities
                                                    # of dark region(s)
                                                    # and 255 showing intensities of bright region(s)
    #print("Intensity threshold valued for quadrant 11:",int(thd11))

    quad = img[0:nr2, nr2:nc]    # quadrant 12
    thd12,thresholded = cv2.threshold(quad, # Image to be segmented
                               0,       # Initial dark region intensity
                               255,     # Initial bright region intensity
                               cv2.THRESH_OTSU)      # Create a binary image with 0 showing intensities
                                                    # of dark region(s)
                                                    # and 255 showing intensities of bright region(s)
    #print("Intensity threshold valued for quadrant 12:",int(thd12))

    quad = img[nr2:nr, 0:nc2]               # quadrant 21
    thd21,thresholded = cv2.threshold(quad, # Image to be segmented
                               0,           # Initial dark region intensity
                               255,         # Initial bright region intensity
                               cv2.THRESH_OTSU)     # Create a binary image with 0 showing intensities
                                                    # of dark region(s)
                                                    # and 255 showing intensities of bright region(s)
    #print("Intensity threshold valued for quadrant 21:",int(thd21))

    quad = img[nr2:nr, nc2:nc]              # quadrant 22
    thd22,thresholded = cv2.threshold(quad, # Image to be segmented
                               0,           # Initial dark region intensity
                               255,         # Initial bright region intensity
                               cv2.THRESH_OTSU)     # Create a binary image with 0 showing intensities
                                                    # of dark region(s)
                                                    # and 255 showing intensities of bright region(s)
    #print("Intensity threshold valued for quadrant 22:",int(thd22))

    # Now segment the original image by adaptive thresholding
    thresholded =img.copy()     # Create segmented image here
    thresholds = img.copy()     # Find threshold value at each pixel and save them here

    # Parameters for bilinear interpolation in quadrant 11
    x22 = (thd11+thd22)/2
    x11 = 2*thd11-x22 
    thd01 = (thd11+thd21)/2 
    thd10 = (thd11+thd12)/2 
    x12 = 2*thd10 - x22 
    x21 = 2*thd01 -x22

    # Additional parameters for bilinear interpolation in quadrant 12
    x13 = 2*thd12-x22
    thd02 = (thd12+thd22)/2 
    x23 = 2*thd02-x22 

    # Additional parameters for bilinear interpolation in quadrant 21
    x31 = 2*thd21-x22
    thd20 = (thd21+thd22)/2
    x32 = 2*thd20-x22

    # Additional parameter for bilinear interpolation in quadrant 22
    x33 = 2*thd22-x22 

    for i in range(nr):
        for j in range(nc):
            if i <= nr2 and j <= nc2:       # Threshold values for the 11 quadrant
                thresholds[i,j] = np.uint8(bilinear(i,j,0,nr2-1,0,nc2-1,x11,x12,x21,x22))
            elif i <= nr2 and j > nc2:    # Threshold values for the 12 quadrant
                thresholds[i,j] = np.uint8(bilinear(i,j,0,nr2-1,nc2,nc-1,x12,x13,x22,x23)) 
            elif i > nr2 and j <= nc2:    # Threshold values for the 21 quadrant
                thresholds[i,j] = np.uint8(bilinear(i,j,nr2,nr-1,0,nc2-1,x21,x22,x31,x32))
            else:                           # Threshold values for the 22 quadrant
                thresholds[i,j] = np.uint8(bilinear(i,j,nr2,nr-1,nc2,nc-1,x22,x23,x32,x33))

    # Now segment the image using the obtained threshold values
    for i in range(nr):
        for j in range(nc):
            if img[i,j] >= thresholds[i,j]:
                thresholded[i,j]=255
            else:
                thresholded[i,j] = 0

    return thresholded

######################### Threshold img using Otsu method #########################
# This method segments a grayscale image by the method of Otsu
# using a single threshold value
def thresholdingOtsu(img):
    thd,image = cv2.threshold(img,  # Image to be segmented
        0,                          # Initial dark region intensity
        255,                        # Initial bright region intensity
        cv2.THRESH_OTSU)            # Create a binary image with 0 showing 
                                    # intensities of dark region(s) and
                                    # 255 showing intensities of bright region(s)
    thd,image = cv2.threshold(img,thd,255,cv2.THRESH_BINARY)
    #print("Optimal threshold value:",int(thd))
    
    return image

######################### Adaptively threshold #########################
# This method segments a grayscale image by adaptive thresholding
def adaptiveThreshold(img):
    image = cv2.adaptiveThreshold(img, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 1)
    
    return image

######################## Find the OR of two binary images ####################
def orImages(img1,img2):
    row1, col1 = img1.shape
    row2, col2 = img2.shape
    img3 = np.zeros((row1,col1), dtype = np.uint8)
    if row1 != row2 or col1 != col2:
        print("Provided images should have the same dimensions.")
    else:
        for i in range(row1):
            for j in range(col1):
                if img1[i,j] == 255 or img2[i,j] == 255:
                    img3[i,j] = 255

    return img3

################################ Inverse the intensities of img #################################
def notImage(img):
    notimg = 255 - img

    return notimg

############################### Find gradients of img ########################
def findGradient(img):
    grad = img.copy()
    rows, cols = grad.shape
    for i in range(rows):
        for j in range(cols):
            if i == rows-1 or j == cols-1:
                grad[i,j] = 0
            else:
                gradi = float(img[i+1,j]) - float(img[i,j])
                gradj = float(img[i,j+1]) - float(img[i,j])
                gr = np.sqrt(gradi*gradi+gradj*gradj)
                if gr>255:
                    gr = 255
                grad[i,j] = int(gr)

    return grad