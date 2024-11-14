# Methods in this module change image intensities to standardize a captured
# image before segmentation. There are other methods that change format
# or intensities of an image for various reasons as commented  before each method.

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv

################ Convert float img to unsigned byte image and return the result ##########
def floatToUbyte(img):
    min = img.min()
    max = img.max()
    rows,cols= img.shape
    image = np.zeros((rows,cols), dtype = np.uint8)
    if max > min:
        scale = 255.0/(max-min)
        for i in range(rows):
            for j in range(cols):
                image[i,j] = int(0.5+(img[i,j]-min)*scale)
    return image

################# Convert BGR color values to HSV color values ##############
# First convert BGR to RGB and then convert RGB to HSV. Replace original
# BGR color components with HSV color components
def convertBGR2HSV(bgrimg, verbose):
    rows,cols,bands = bgrimg.shape
    rgb = bgrimg.copy()
    for i in range(rows):
        for j in range(cols):
            (b,g,r) = bgrimg[i,j,:]
            rgb[i,j,:] = (r,g,b)
    hsv = rgb2hsv(rgb)
    hue = np.zeros((rows,cols), dtype=float)
    sat = np.zeros((rows,cols), dtype=float)
    val = np.zeros((rows,cols), dtype=float)
    for i in range(rows):
        for j in range(cols):
            h = hsv[i,j,0]
            hue[i,j] = h 
            s = hsv[i,j,1]
            sat[i,j] = s  
            v = hsv[i,j,2]
            val[i,j] = v  
    
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4, figsize=(10, 2))

    ax0.imshow(rgb)
    ax0.set_title("RGB")
    ax0.axis('off')
    hue = floatToUbyte(hue)
    ax1.imshow(hue, cmap = 'gray')
    ax1.set_title("Hue")
    ax1.axis('off')
    sat = floatToUbyte(sat)
    ax2.imshow(sat, cmap = 'gray')
    ax2.set_title("Saturation")
    ax2.axis('off')
    val = floatToUbyte(val)
    ax3.imshow(val, cmap = 'gray')
    ax3.set_title("Value")
    ax3.axis('off')
    fig.tight_layout()

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(10, 2))

    ax0.hist(hue.ravel(), 256)
    ax0.set_title("Histogram of Hue")
    ax1.hist(sat.ravel(), 256)
    ax1.set_title("Histogram of Saturation")
    ax2.hist(val.ravel(), 256)
    ax2.set_title("Histogram of Value")
    fig.tight_layout()
    if verbose: 
        plt.show()

    return hue, sat, val

####################### Dull shiny spots in a grayscale image ####################
# Find square-root of image intensities, stretch intensities to 0-255, and
# replace the original intensities with new intensities.
def transformSqrtIntensities(img):
    rows, cols = img.shape
    dimg = np.empty([rows,cols], dtype = float)
    dullimg = img.copy()
    mn = 255.0 #img.max()-img.min()
    for i in range(rows):
        for j in range(cols):
            dimg[i,j] = math.sqrt(img[i,j])
    min = dimg.min()
    max = dimg.max()
    for i in range(rows):
        for j in range(cols):
            d = (dimg[i,j]-min)*mn/(max-min)
            dullimg[i,j] =round(d)
    #print(dullimg.min(), dullimg.max())
    return dullimg

####################### Dull shiny spots in a color image ####################
# Find square-root of color components, stretch values to 0-255, and
# replace the original colors with new colors.
def transformSqrtColors(img):
    rows, cols, bands = img.shape
    dimg = np.empty([rows,cols,bands], dtype = float)
    dullimg = img.copy()
    mn = 255.0      # Max value of color image
    for i in range(rows):
        for j in range(cols):
            for k in range(bands):
                dimg[i,j,k] = math.sqrt(img[i,j,k])
    min = dimg.min()
    max = dimg.max()
    for i in range(rows):
        for j in range(cols):
            for k in range(bands):
                d = (dimg[i,j,k]-min)*mn/(max-min)
                dullimg[i,j,k] =round(d)
    #print(dullimg.min(), dullimg.max())
    return dullimg

