# This module contains methods to load, save, and display images
# Note use of cv2 in these methods; therefore, loaded color images are in BGR format.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import TransformGeometry as tg

def loadGrayImage(imgname):
    img = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(imgname,"could not be found or read.")
        exit(0)
    return img

################################## Load an image ############################
def loadImage(imgname, color):
    cimg = cv2.imread(imgname, color)   # color = 1 implies load image as color (BGR) 
    if cimg is None:
        print(imgname,"could not be found or read.")
        exit(0)

    gray = True
    if cimg.ndim == 3:
        nr, nc, _ = cimg.shape
        for i in range(nr):
            for j in range(nc):
                if cimg[i,j,0] != cimg[i,j,1] or cimg[i,j,0] != cimg[i,j,2] or cimg[i,j,1] != cimg[i,j,2]:
                    gray = False

    if gray:
        cimg = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE)
        
    return cimg

############################ display an image ############################
# Display a gray scale or color image
# Remove the display if any key on keyboard is pressed
# If key is 's', save the image in jpg format in file with provided name
def displayImage(name,img):
    cv2.imshow(name, img)
    k = cv2.waitKey(0)
    if k == 115:        # Save the image if key 's' is pressed
        cv2.imwrite(name+".jpg",img)
        print("Saved image in file "+name+".jpg")    

############################ display a smaller image ############################
# Display a gray scale or color image. If image is too large, scale down
# image appropriately. Remove the display if any key on keyboard is pressed.
def displayResizedImage(title,img, size, location):
    import pyautogui
    
    scwidth,scheight = pyautogui.size()
    
    dim = img.ndim
    if dim == 2:
        nr, nc = img.shape
    else:
        nr, nc, _ = img.shape
    while nr*nc > size:
        nr = nr//2
        nc = nc//2
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, nc, nr)
    cv2.imshow(title, img)
    
    cloc = location[0]
    rloc = location[1]
    cloc += nc//4
    if cloc+nc > scwidth:
        cloc = 2
        rloc += nr//4
        if rloc+nr > scheight:
            cloc = 2
            rloc = 2
    location[0] = cloc
    location[1] = rloc
    cv2.moveWindow(title, cloc, rloc)

    cv2.waitKey(0)

############################ Display scaled (up or down) img ###########################
def displayScaledImage(title,img, sc, location):
    import pyautogui
    scwidth,scheight = pyautogui.size()

    dim = img.ndim
    nr = 0
    nc = 0
    if dim == 2:
        nr,nc = img.shape
    else:
        nr,nc,_ = img.shape
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, nc*sc, nr*sc)
    cv2.imshow(title, img)
    cloc = location[0]
    rloc = location[1]
    cloc += nc*sc
    if cloc+nc*sc > scwidth:
        cloc = 2
        rloc += nr*sc
        if rloc+nr*sc > scheight:
            cloc = 2
            rloc = 2
    location[0] = cloc
    location[1] = rloc
    cv2.moveWindow(title, cloc, rloc)
    cv2.waitKey(0)
    

############################## Display a list of images ################################
# Display a list of images
# To remove the displayed images, press any key on keyboard
def displayImages(title,images):
    import pyautogui
    
    scwidth, scheight = pyautogui.size()

    n = len(images)
    dim = 2
    nr = 0
    nc = 0
    if n > 0:
        cloc = 2
        rloc = 2
        maxvs = 0       # maximum vertical spacing between image rows
        for i in range(n):
            dim = images[i].ndim
            if dim == 2:
                nr,nc = images[i].shape
            else:
                nr,nc,_ = images[i].shape
            if nr > maxvs:
                maxvs = nr
            cv2.namedWindow(f"{title}{i}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{title}{i}", nc, nr)
            cv2.imshow(f"{title}{i}",images[i])
            
            cloc += nc+20
            if cloc+nc+20 > scwidth:
                cloc = 2
                rloc += maxvs+30
                maxvs = 0
                if rloc+nr+30 > scheight:
                    cloc = 2
                    rloc = 2
                    maxvs = 0
            cv2.moveWindow(f"{title}{i}", cloc, rloc)

        cv2.waitKey(0)

# Display a list of labeled coins
# To remove the images press any key
def displayLabeledCoins(titles,images, coinTypes):
    import pyautogui
    
    scwidth,scheight = pyautogui.size()

    n = len(images)
    dim = 2
    nr = 0
    nc = 0
    if n > 0:
        cloc = 2
        rloc = 2
        maxvs = 0       # maximum vertical spacing between image rows
        for i in range(n):
            title = f"{i}:{coinTypes[titles[i]]}"
            print(title)
            dim = images[i].ndim
            if dim == 2:
                nr,nc = images[i].shape
            else:
                nr,nc,_ = images[i].shape
            nr = 2*nr
            nc = 2*nc
            if nr > maxvs:
                maxvs = nr
            
            
            cloc += nc
            if cloc+nc > scwidth:
                cloc = 2
                rloc += maxvs+30
                maxvs = 0
                if rloc+nr+30 > scheight:
                    cloc = 2
                    rloc = 2
                    maxvs = 0
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, nc, nr)
            cv2.imshow(title,images[i])
            cv2.moveWindow(title, cloc, rloc)

        cv2.waitKey(0)
        
def displayScaledImages(title,images, sc):
    import pyautogui
    
    scwidth,scheight = pyautogui.size()

    n = len(images)
    dim = 2
    nr = 0
    nc = 0
    if n > 0:
        cloc = 2
        rloc = 2
        maxvs = 0       # maximum vertical spacing between image rows
        for i in range(n):
            dim = images[i].ndim
            if dim == 2:
                nr,nc = images[i].shape
            else:
                nr,nc,_ = images[i].shape
            nr = nr*sc
            nc = nc*sc
            if nr > maxvs:
                maxvs = nr
            cv2.namedWindow(f"{title}{i}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"{title}{i}", nc, nr)
            cv2.imshow(f"{title}{i}",images[i])
            
            cloc += nc+20
            if cloc+nc+20 > scwidth:
                cloc = 2
                rloc += maxvs+30
                maxvs = 0
                if rloc+nr+30 > scheight:
                    cloc = 2
                    rloc = 2
                    maxvs = 0
            cv2.moveWindow(f"{title}{i}", cloc, rloc)

        k = cv2.waitKey(0)

########################## Save a list of images #############################
# Given a list of images and a filename, save the images individually in 
# a directory created from filename
def saveImages(filename,images):
    n = len(images)
    if n == 0:
        print("No images saved!")
    else:
        basename = os.path.basename(filename) # get basename
        dir = os.path.dirname(filename)       # get dir name
        parts = basename.rsplit(".")          # get image format
        name = parts[0]   
        fmt = "png" #parts[1]   
        if dir == "":
            dir = name
        else:
            dir = dir+"/"+name
        if not os.path.exists(dir):     # If the dir does not exist
            os.mkdir(dir)               # create it; otherwise, use it  
        for i in range(n):
            nname = name+f"{i}."+fmt    # Add to base name, image number, a dot, and format
            dirpath = os.path.join(dir,nname)   # Concatenate dir and nname
            cv2.imwrite(dirpath,images[i])      # Save image i at dirpath
        print(f"{n} images saved in directory {dir}.")

######################## Convert PIL image to OpenCV image ####################
def convertPIL2Opencv(img):
    nimg = np.array(img)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    return nimg

######################## Convert OpenCV image to PIL image ####################
def convertOpencv2PIL(img):
    nimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    nimg = Image.fromarray(nimg)
    return nimg