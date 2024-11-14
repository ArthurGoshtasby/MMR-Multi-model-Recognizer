# This module contains method to process image controus, some of
# which end up as detected coin boundaries.

import cv2
import numpy as np
import math 


minlength = 100     # minimum contour length to consider it a coin boundary
bbval = 128         # intensity of bounding boxes of detected coins
coindiameter = 129  # diameter of standardized coin (radius=64)
maxcoins = 254      # max number of coins that can be detected in an image

########################## Find region boundaries ##########################
# Find boundaries of bright regions in a dark background in provided image
def findBoundaries(image):
    rows,cols = image.shape
    img = np.zeros((rows,cols),dtype = np.uint8)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if image[i,j] == 255:
                if image[i,j+1] == 0 or image[i+1,j] == 0 \
                    or image[i,j-1] == 0 or image[i-1,j] == 0 \
                    or image[i+1,j+1] == 0  or image[i+1,j-1] == 0 \
                    or image[i-1,j-1] == 0  or image[i-1,j+1] == 0:
                    img[i,j] = 255

    return img


################################ Is point inside contour? ######################
# See if point is inside contour
def isinside(point,contour):
    (r,c) = point
    # Find two points (r,j1) and (r,j2) in contour such that 
    # j1 < c < j2.
    j1 = -1 
    j2 = -1
    for k in range(len(contour)):
        (i,j) = contour[k]
        if i == r:
            if j < c:
                j1 = j
            elif j > c:
                j2 = j
    if j1 >= j2: 
        return False

    # If the above is True, find two points (i1,c) and (i2,c) in contour where
    # i1 < r < i2
    i1 = -1
    i2 = -1
    for k in range(len(contour)):
        (i,j) = contour[k]
        if j == c:
            if i < r:
                i1 = i
            elif i > r:
                i2 = i
    if i1 >= i2:
        return False
    
    # If none of the above, then point is inside contour.
    return True

############### Remove contours falling inside other contours ###############
def removeInteriorContours(contours):
    global minlength
    newcontours = []
    n = len(contours)   # number of regions
    inside = np.zeros(n)
    
    for i in range(n-1):
        contouri = contours[i]
        for j in range(i+1,n):
            contourj = contours[j]
            if isinside(contouri[10],contourj): 
                inside[i] = 1
            elif isinside(contourj[10],contouri):
                inside[j] = 1 
    for i in range(n):
        if inside[i] == 0:
            newcontours.append(contours[i])

    return newcontours

# Find the boundary point next to point in image and return it
def getNextPoint(point,img):
    rows, cols = img.shape
    (r,c) = point
    npoint = ()
    if c+1 < cols and img[r,c+1] == 255:
        npoint = (r,c+1)
    elif r+1 < rows and img[r+1,c] == 255:
        npoint =(r+1,c)
    elif c-1 >= 0 and img[r,c-1] == 255:
        npoint = (r,c-1)
    elif r-1 >= 0 and img[r-1,c] == 255:
        npoint =(r-1,c)
    elif r+1 < rows and c+1 < cols and img[r+1,c+1] == 255:
        npoint = (r+1,c+1)
    elif r+1 < rows and c-1 >= 0 and img[r+1,c-1] == 255:
        npoint = (r+1,c-1)
    elif r-1 >= 0 and c-1 >= 0 and img[r-1,c-1] == 255:
        npoint = (r-1,c-1)
    elif r-1 >= 0 and c+1 < cols and img[r-1,c+1] == 255:
        npoint = (r-1,c+1)

    return npoint


############################ Find points along a boundary ########################
# Find a closed boundary within img and return points along it
def findBoundaryPoints(img,point):
    (i,j) = point
    bpts = []
    img[i,j] = 0
    bpts.append(point)
    point = getNextPoint(point, img)
    while point != ():
        (r,c) = point
        img[r,c] = 0
        bpts.append(point)
        point = getNextPoint(point,img)

    return bpts

################################ Find contours ############################
# Create a list of pixels along boundary contours in an image containing boundaries of
# regions in a segmented image. If the number of pixels along a contour is 
# fewer than minlength, ignore the contour. Otherwise, add the contour to
# the list of contours and return the list upon completion.
def findContours(image):
    global minlength
    img = image.copy()
    rows, cols = img.shape
    contours = []
    # Find points along a boundary
    done = False
    while not done:
        count = 0
        done = True
        for i in range(rows):
            for j in range(cols):       
                if img[i,j] == 255:
                    point = (i,j)
                    contour = findBoundaryPoints(img,point)    
                    if len(contour) > minlength:
                        (r1,c1) = contour[0]    # make sure contour is closed
                        (r2,c2) = contour[-1]
                        if abs(r1-r2) < 2 and abs(c1-c2) < 2:
                            contours.append(contour)
                            done = False
                        else:
                            count += 1
                            if count < rows+cols:
                                # Put contour points back in image and try again
                                for k in range(len(contour)):
                                    (r,c) = contour[k]
                                    img[r,c] = 255
                                done = False
                            else:
                                # Try no more! contour is malformed
                                done = False
                                print("Encountered a malformed region, skipping it.")
    return contours

######################## Add contours to image ################################
# Draw detected contour over image for visual examination of detected contours
def drawBoundaries(image,contours):
    img = image.copy()
    dim = img.ndim
    
    if dim == 3:
        rows, cols, bands = img.shape
        img = np.zeros((rows,cols,bands), dtype=np.uint8)
        for i in range(len(contours)):
            contour = contours[i]
            for j in range(len(contour)):
                (r,c) = contour[j]
                if r >=0 and r <rows and c >=0 and c < cols:
                    img[r,c,0] = 255
                    img[r,c,1] = 0
                    img[r,c,2] = 0
    else:
        rows,cols = img.shape
        img = np.zeros((rows,cols), dtype=np.uint8)
        for i in range(len(contours)):
            contour = contours[i]
            for j in range(len(contour)):
                (r,c) = contour[j]
                if r >=0 and r <rows and c >=0 and c < cols:
                    img[r,c] = 255

    return img

####################### Draw exterior boundaries ##########################
# Draw only the exterior contour boundaries, ignoring possible interior contours
def drawExteriorBoundaries(image,contours):
    img = image.copy()
    dim = img.ndim
    
    if dim == 3:
        img[:,:,:] = 0 #clear img
        rows, cols, bands = img.shape
        for i in range(len(contours)):
            contour = contours[i]
            for j in range(len(contour)):
                (r,c) = contour[j]
                if r >=0 and r <rows and c >=0 and c < cols:
                    img[r,c,0] = 255
                    img[r,c,1] = 255
                    img[r,c,2] = 255
    else:
        rows,cols = img.shape
        img[:,:] = 0 #clear img
        for i in range(len(contours)):
            contour = contours[i]
            for j in range(len(contour)):
                (r,c) = contour[j]
                if r >=0 and r <rows and c >=0 and c < cols:
                    img[r,c] = 255

    return img

################## Find bounding boxes of contours #######################
# Find bounding boxes of contours and return the coordinates of
# the ulhc and lrhc of each contour in list bboxes. This is a means
# to extract individual coins from an image.
def findBoundingBoxes(contours):
    bboxes = []
    for i in range(len(contours)):
        contour = contours[i]
        if len(contour) == 0:
            print(f"contour {i} is empty")
        minr = 9999
        minc = 9999
        maxr = -1
        maxc = -1

        for j in range(len(contour)):
            (r,c) = contour[j]
            if r>maxr:
                maxr = r
            elif r<minr:
                minr = r
            if c>maxc:
                maxc = c
            elif c<minc:
                minc=c
        bbox = (minr,minc,maxr,maxc)
        bboxes.append(bbox)

    return bboxes

###################### Draw the bounding boxes in image ########################
# Draw the bounding boxes in image and return the image. This is primarily 
# for the purpose of better visually examining detected coins in image
def drawBoundingBoxes(image,bboxes):
    global bbval
    img = image.copy()
    dim = img.ndim
    for k in range(len(bboxes)):
        (r1,c1, r2,c2) = bboxes[k]
        if dim == 3:
            for i in range(r1,r2+1):
                if i % 2 == 0:
                    img[i,c1,:] = 0
                    img[i,c2,:] = 0
                else:
                    img[i,c1,:] = bbval
                    img[i,c2,:] = bbval
            for j in range(c1,c2+1):
                if j % 2 == 0:
                    img[r1,j,:] = 0
                    img[r2,j,:] = 0
                else:
                    img[r1,j,:] = bbval
                    img[r2,j,:] = bbval
        else:
            for i in range(r1,r2+1):
                if img[i,c1] == 0:
                    img[i,c1] = bbval
                if img[i,c2] == 0:
                    img[i,c2] = bbval
            for j in range(c1,c2+1):
                if img[r1,j] == 0:
                    img[r1,j] = bbval
                if img[r2,j] == 0:
                    img[r2,j] = bbval

    return img

##################### Fill the boundary within bbox with label #################
# Fill the closed boundary enclosed by bbox within img with
# intensity equal to the provided label.
def fillAndLabelBoundary(img,bbox,label):
    rows,cols = img.shape
    (r1,c1,r2,c2) = bbox
    lpoints = set()  # set of labeled points 
    bpoints = set()  # set of boundary points
    rpoints = set()  # set of region points
    # Find a point inside the boundary: (r,c)
    r = (r1+r2)//2
    c = (c1+c2)//2
    # Fill region with point (r,c) in it
    img[r,c]=label
    lpoints.add((r,c))
    rpoints.add((r,c))
    while len(lpoints) > 0:
        r,c = lpoints.pop()
        if r<0 or c<0 or r>rows-1 or c>cols-1:
            continue
        if r > 0:
            if img[r-1,c] == 0: 
                img[r-1,c] = label
                lpoints.add((r-1,c))
                rpoints.add((r-1,c))
            elif img[r-1,c] == 255:
                bpoints.add((r-1,c))
        if r < rows-1:
            if img[r+1,c] == 0:
                img[r+1,c] = label
                lpoints.add((r+1,c))
                rpoints.add((r+1,c))
            elif img[r+1,c] == 255:
                bpoints.add((r+1,c))
        if c > 0:
            if img[r,c-1] == 0:
                img[r,c-1] = label
                lpoints.add((r,c-1))
                rpoints.add((r,c-1))
            elif img[r,c-1] == 255:
                bpoints.add((r,c-1))
        if c < cols-1:
            if img[r,c+1] == 0:
                img[r,c+1] = label
                lpoints.add((r,c+1))
                rpoints.add((r,c+1))
            elif img[r,c+1] == 255:
                bpoints.add((r,c+1))
    
        # Add boundary points to region
        while len(bpoints) > 0:
            (r,c) = bpoints.pop()
            img[r,c] = label
            rpoints.add((r,c))

    return len(rpoints)

################### Suppress background while labeling contours ################
# Given img that contains region boundaries and a list of bboxes containing 
# the boundaries, this method fills the boundaries with intensities showing
# their labels.
def suppressBackground(img,bboxes):
    global maxcoins

    rows = 0
    cols = 0
    ndim = img.ndim
    if ndim == 3:
        print("The image passed to method suppressBackground() should be grayscale.")
        exit(0)
    else:
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if img[i,j] > 0 and img[i,j] < 255:
                    print("The image passed to method suppressBackground() should be binary, showing region baoundaries.")
                    exit(0)
    bgimg = img.copy()
    labels = []
    nofc = len(bboxes)
    inc = 1
    if nofc > maxcoins:
        print(" Too many boundaries, only 254 boundaries will be filled and labeled.")
    else:
        inc = maxcoins//(nofc+1)

    for k in range(len(bboxes)):
        label = (k+1)*inc
        nrp = fillAndLabelBoundary(bgimg,bboxes[k],label)
        if nrp > 0:
            labels.append((label,nrp))

    return labels, bgimg

# Given original image, labeled lblimg, and 
# bboxes of regions, extract the regions and return them in a list
def extractRegions(image,lblimg,bboxes):
    regions = []
    ndim = image.ndim
    for k in range(len(bboxes)):
        img = []
        (r1,c1,r2,c2) = bboxes[k]
        r = (r1+r2)//2
        c = (c1+c2)//2
        lbl = lblimg[r,c]
        nr = r2-r1+1 
        nc = c2-c1+1 
        if ndim == 3:
            img = np.zeros((nr,nc,ndim), dtype=np.uint8)
            for i in range(nr):
                for j in range(nc):
                    if lblimg[i+r1,j+c1] == lbl:
                     img[i,j,:] = image[i+r1,j+c1,:]
        else:
            img = np.zeros((nr,nc), dtype=np.uint8)
            for i in range(nr):
                for j in range(nc):
                    if lblimg[i+r1,j+c1] == lbl:
                        img[i,j] = image[i+r1,j+c1]
        regions.append(img)

    return regions


# Find Euclidean distance between two points
def dist(p1,p2):
    (y1,x1) = p1
    (y2,x2) = p2
    dy = y2-y1
    dx = x2-x1
    d =math.sqrt(dx*dx+dy*dy)

    return d

# Find major and minor axes endpoints of the ellipse representing 
# the contour.
def findShape(contour):
    shape = [] 
    ellipse = True
    n = len(contour)
    # Find center of shape
    cx = 0
    cy = 0
    for i in range(n):
        (y,x) = contour[i]
        cx += x
        cy += y

    cx = int(cx/n)
    cy = int(cy/n)
    cp = (cy,cx)
    
    # get a histogram of radii
    r = cy*2
    if cx > cy:
        r = cx*2
    hist = np.zeros((r,), dtype=np.int16)
    for i in range(n):
        d = dist(contour[i],cp)
        j = int(d)
        hist[j] += 1

    # See if the region is circular or elliptic
    minr=0
    maxr=0
    for j in range(r):
        if minr == 0 and hist[j] == 0:
            continue
        elif minr == 0 and hist[j] > 0:
            minr = j
        elif minr > 0 and hist[j] > 0:
            continue
        elif minr > 0 and hist[j] == 0:
            maxr = j-1
            break
    ratio = float(minr)/float(maxr)
    radius = 0
    count = 0
    if ratio > 0.9:      # We have a circle
        ellipse = False
        # radius of circle is the radius most frequently occurring
        for j in range(r):
            if hist[j] > count:
                count = hist[j]
                radius = j
        shape.append(cp)
        shape.append(radius)
    else:              # We have an ellipse
        # Find contour point farthest from the center
        dmax = 0
        k1 = 0
        for k in range(n):
            d = dist((cy,cx),contour[k])
            if d>dmax:
                dmax = d
                k1 = k

        # Find opposing point on contour
        if k1 < n//2:
            k2 = k1+n//2
        else:
            k2 = k1-n//2
        # k1 and k2 points define major axis

        # Find minor-axis endpoints
        k3 = (k1+k2)//2
        
        k4 = k3 + n//2
        if k4 > n:
            k4 = k4 - n
        shape.append(contour[k1])
        shape.append(contour[k2])
        shape.append(contour[k3])
        shape.append(contour[k4])

    return ellipse, shape

# Find major and minor axes endpoints of ellipses representing 
# the contours.
def findShapes(contours):
    ellipses = [] 
    circles = []
    for k in range(len(contours)):
        contour = contours[k]
        ellipse, shape = findShape(contour)
        shape.append(k)
        if ellipse:
            ellipses.append(shape)
        else:
            circles.append(shape)

    return circles, ellipses

# Create a contour representing a connected circle of a given center and radius
def drawCircle(center,radius):
    import math
    (r,c) = center
    contour = []
    # There are 2*PI*radius points along circle
    # Draw the points with angular increment of 2*PI/2*PI*radius or 1/radius
    dtheta = 1.0/radius 
    theta = 0.0
    while theta < 2.0*math.pi:
        rr = r+radius*math.sin(theta)
        cc = c+radius*math.cos(theta)
        point = (int(rr),int(cc))
        contour.append(point)
        theta += dtheta

    return contour

# Find points in contour of distance rad to center
def findPoints(contour,center,rad):
    points = []
    (r,c) = center
    for k in range(len(contour)):
        (rr,cc) = contour[k]
        dr = rr-r
        dc = cc-c
        d = np.sqrt(dr*dr+dc*dc)
        if rad == int(d):
            points.append(contour[k])

    return points

# Find radius of contour with given center
# Consider best radius to be the one representing distance of most 
# contour points to center
def findRadius(contour,center):
    (r,c) = center
    hist = np.zeros(1000, dtype=int)
    for k in range(len(contour)):
        (rr,cc) = contour[k]
        # find distance of (r,c) and (rr,cc)
        dr = rr-r
        dc = cc-c
        d = np.sqrt(dr*dr+dc*dc)
        # Increment entry of histogram at radius d by 1
        hist[int(d)] += 1

    # find histogram entry with max value
    count = hist[0]
    radius = 0
    for i in range(1,500):
        if hist[i] > count:
            count = hist[i]
            radius = i
    if count*20 < len(contour): # If contour is not circular keep it as is
        print("Encountered a non-circular region.")
        return -1
    else: 
        return radius 

# Find coordinates of the center of contour
def findCenter(contour):
    nr = 0
    nc = 0
    for k in range(len(contour)):
        (r,c) = contour[k]
        nr += r
        nc += c

    r = nr//k
    c = nc//k

    return (r,c)

########################## Find the circle best fitting a contour ###################
# If a contour is supposed to represent a circular region, find the best
# circle fitting it and replace the contour with the circle
def getBestFittingCircularContours(contours):
    from circle_fit import standardLSQ
    ncontours = []
    for k in range(len(contours)):
        contour = contours[k]
        center = findCenter(contour)
        radius = findRadius(contour,center)
        if radius < 10:
            continue
        points = findPoints(contour,center,radius)
        if len(points) < 10:
            continue
        cr, cc, radius, err = standardLSQ(points)
        #print(f"Center: ({int(cr)},{int(cc)}), radius: {int(radius)}, err: {err:.4f}")
        center = (cr,cc)

        if err < 0.5:
            circle = drawCircle(center,radius-5) # Don't draw the coin edges
            ncontours.append(circle)
        else:
            print("Encountered a non-circular contour.")
            #ncontours.append(contour)

    return ncontours
