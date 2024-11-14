
import numpy as np
import cv2
from PIL import Image
import LoadDisplay as ld
import math

def findAffine(refs,tars):
    # The parameters to be estimated
    x = np.ones(6, dtype  = np.float32)
    if len(refs) != len(tars):
        print("Corresponding reference and target points not provided.")
        return False, []
    r = len(refs)
    A = np.ones((r,3), dtype = np.float32)
    for i in range(r):
        for j in range(2):
            A[i,j] = refs[i][j]

    b = np.ones(r, dtype = np.float32)
    for i in range(r):
        b[i] = tars[i][0]

    # Find x-component transformation
    try:
        x1 = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.linalg.LinAlgError:
        print("Singular matrix of coefficients.")
        return False, []

    # Find y-component transformation
    for i in range(r):
        b[i] = tars[i][1]
    
    try:
        x2 = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.linalg.LinAlgError:
        return False, []    # Singular matrix of coefficients

    for i in range(3):
        x[i] = x1[i]
        x[i+3] = x2[i]
    M = x.reshape(2,3)
    return True, M          # return affine transformation matrix

def getAffinePoint(refp, M):
    r = int(M[0][0]*refp[0]+M[0][1]*refp[1]+M[0][2]+0.5)
    c = int(M[1][0]*refp[0]+M[1][1]*refp[1]+M[1][2]+0.5)
    return (r,c)

def extractRegion(image):
    maxr = 0
    maxc = 0
    minr = 9999
    minc = 9999
    dim = image.ndim
    img = []
    if dim == 3:
        rows,cols, bands = image.shape
        for i in range(rows):
            for j in range(cols):
                if image[i,j,0]>0 or image[i,j,1]>0 or image[i,j,2]>0:
                    if i>maxr:
                        maxr = i
                    elif i<minr:
                        minr = i
                    if j>maxc:
                        maxc = j
                    elif j<minc:
                        minc = j
        print(minr,minc,maxr,maxc)
        w = maxc-minc+1
        h = maxr-minr+1
        print(h,w)
        if h<1 or w<1:
            print("Empty image!")
            img = image.copy()
            return img
        img = np.zeros((h,w,bands), dtype=np.uint8)
        for i in range(minr,maxr+1):
            r =i-minr
            for j in range(minc,maxc+1):
                c = j-minc
                img[r,c,:]=image[i,j,:]
    else:
        rows,cols = image.shape
        for i in range(rows):
            for j in range(cols):
                if image[i,j]>0:
                    if i>maxr:
                        maxr = i
                    elif i<minr:
                        minr = i
                    if j>maxc:
                        maxc = j
                    elif j<minc:
                        minc = j
        print(minr,minc,maxr,maxc)
        w = maxc-minc+1
        h = maxr-minr+1
        print(h,w)
        if h<1 or w<1:
            print("Empty image!")
            img = image.copy()
            return img
        img = np.zeros((h,w), dtype=np.uint8)
        for i in range(minr,maxr+1):
            r =i-minr
            for j in range(minc,maxc+1):
                c = j-minc
                img[r,c]=image[i,j]

    return img 

def affineTransform(src,M,drows):
    dcols = drows
    dim = src.ndim
    dst = np.zeros((drows,dcols),dtype=np.uint8)

    if dim == 3:
        (rows,cols,bands) = src.shape
        dst = np.zeros((drows,dcols,bands), dtype=np.uint8)
        for r in range(drows):
            for c in range(dcols):
                (rr,cc) = getAffinePoint((r,c),M)
                if rr>=0 and rr<rows and cc>=0 and cc<cols:
                    dst[r,c,:]=src[rr,cc,:]
    else:
        (rows,cols) = src.shape
        for r in range(drows):
            for c in range(dcols):
                (rr,cc) = getAffinePoint((r,c),M)
                if rr>=0 and rr<rows and cc>=0 and cc<cols:
                    dst[r,c]=src[rr,cc]
    dst = extractRegion(dst)

    return dst

# Given src image with circle at scenter and radius sr
# Rescale it to a circle of radius dr centered at dst image
# of size dwidthxdwidth.
def rescale(src,sr,dwidth,dr):
    dim = src.ndim
    scale = float(sr)/float(dr)
    dst = [] 
    if dim ==3:
        rows,cols,bands = src.shape
        scr = rows//2
        scc = cols//2
        dst = np.zeros((dwidth,dwidth,bands), dtype = np.uint8)
        for i in range(dwidth):
            r = int((i-dwidth//2)*scale+scr+0.5)
            for j in range(dwidth):
                c = int((j-dwidth//2)*scale+scc+0.5)
                if r>=0 and r<rows and c>=0 and c<cols:
                    dst[i,j,:] = src[r,c,:]
    else:
        rows,cols = src.shape
        scr = rows//2
        scc = cols//2
        dst = np.zeros((dwidth,dwidth), dtype = np.uint8)
        for i in range(dwidth):
            r = int((i-dwidth//2)*scale+scr+0.5)
            for j in range(dwidth):
                c = int((j-dwidth//2)*scale+scc+0.5)
                if r>=0 and r<rows and c>=0 and c<cols:
                    dst[i,j] = src[r,c]
    # Extract the center part of dst containing the coin
    dst = extractRegion(dst)  
    
    return dst

############################## Resize image ##################################
# Resize src to a square image of side equal to width and return it.
def resize(src,width):
    dim = src.ndim

    if dim ==3:
        rows,cols,bands = src.shape
        dst = np.zeros((width,width,bands), dtype = np.uint8)
        scr = float(rows)/float(width)
        scc = float(cols)/float(width)
        for i in range(width):
            r = int(i*scr+0.5)
            for j in range(width):
                c = int(j*scc+0.5)
                if r>=0 and r<rows and c>=0 and c<cols:
                    dst[i,j,:] = src[r,c,:]
    else:
        rows,cols = src.shape
        dst = np.zeros((width,width), dtype = np.uint8)
        scr = float(rows)/float(width)
        scc = float(cols)/float(width)
        for i in range(width):
            r = int(i*scr+0.5)
            for j in range(width):
                c = int(j*scc+0.5)
                if r>=0 and r<rows and c>=0 and c<cols:
                    dst[i,j] = src[r,c]
    
    return dst

# Rotate src (in PIL format) by angle theta in degrees and return it
# to be in PIL format.
def rotatePILImage(src, theta):
    dst = src.rotate(theta)

    return dst

#################################### Rotate image ###########################
# Rotate src (in opencv format) by angle theta in degrees and return it.
def rotateImage(src, theta):
    dim = src.ndim
    dst = []
    if dim == 2:        # If image is grayscale
        # First convert Opencv format to PIL format
        dst = Image.fromarray(src)
        # Do the rotation
        dst = dst.rotate(theta) 
        # Then convert PIL format back to Opencv format
        dst =np.array(dst)
    else:               # If image is color
        dst = ld.convertOpencv2PIL(src)
        # Do the rotation
        dst = dst.rotate(theta) 
        dst = ld.convertPIL2Opencv(dst)

    return dst

# Find most dominant orientations of coin and return the first nor orientations
# If only nor > n (all dominant orientations of coin), just return the n orientations found.
def findDominantOrientations(coin, nor, spacing = 10):
    lor = []            # list of dominant orientations of coin in degrees
    loc = []            # and their locations
    n = 0               # number of orientations found
    PI = math.pi
    sp = spacing        # default minimum spacing between peaks
    # Map coin intensities radially to its circular boubdary.
    # Weigh an intensity by its distance to center of coin.
    dim = coin.ndim
    
    rows = 0
    cols = 0
    # Use gradients (rather than intensities) to determine dominant orientations
    if dim == 2:
        rows, cols = coin.shape
        coinn = cv2.GaussianBlur(coin, (3,3), 0)
        coinn = cv2.Laplacian(coinn, cv2.CV_8U, ksize=3)
    else:
        rows, cols, _ = coin.shape
        coin0 = coin[:,:,0]
        coin1 = coin[:,:,1]
        coin2 = coin[:,:,2]
        coin0 = cv2.GaussianBlur(coin0, (3,3), 0)
        coin1 = cv2.GaussianBlur(coin1, (3,3), 0)
        coin2 = cv2.GaussianBlur(coin2, (3,3), 0)
        coin0 = cv2.Laplacian(coin0, cv2.CV_8U, ksize=3)
        coin1 = cv2.Laplacian(coin1, cv2.CV_8U, ksize=3)
        coin2 = cv2.Laplacian(coin2, cv2.CV_8U, ksize=3)
        coinn = coin.copy()
        coinn[:,:,0]=coin0
        coinn[:,:,1]=coin1
        coinn[:,:,2]=coin2
    r = (rows-1)//2                 # radius of coin
    cr = cc = r                     # coordinates of center of coin

    p = 2.0*PI*(r-1)                    # perimeter of coin (not including border pixels)
    m = int(p)                          # number of points along the boundary
    bound = np.zeros(m, dtype = int)
    for i in range(m):
        theta = i*2*PI/m
        for j in range(r-1):            # don't use border pixels
            x = cc + int(float(j)*math.cos(theta))
            y = cr + int(float(j)*math.sin(theta))
            y = rows-1-y
            if y < 0 or x < 0 or y >= rows or x >= cols:
                print(f"x={x}, y={y}")
                continue
            if dim == 2:
                bound[i] += int(coinn[y,x])*j       # increment boundary point by pixel intensity
                                                    # times its distance to coin center
            else:
                bound[i] += j*(int(coinn[y,x,0])+int(coinn[y,x,1])+int(coinn[y,x,2]))
                
    # smooth the boundary values to avoid noisy peaks
    b = np.zeros(m, dtype = int)
    b[:] = bound[:]
    for i in range(m):
        bound[i]=0
        for j in [-1,0,1]:
            if j+i<0:
                bound[i] += b[j+i+m]
            elif j+i >= m:
                bound[i] += b[j+i-m]
            else:
                bound[i] += b[j+i]
                
    # Find point on the boundary with the highest value. That point
    # shows the location of the most dominant orientation of coin.             
    t = 0                                           # angle of dominant orientation
    maxval = np.zeros(nor,dtype = int)              # and its corresponding value
    ii = 0
    for i in range(m):
        if bound[i] > maxval[0]:
            maxval[0] = bound[i]      # remember value at dominant orientation
            ii = i                    # remember location of dominant orientation
            t = int(i*360.0/m+0.5)    # remember angle of dominant orientation in degrees
    lor.append(t)
    loc.append(ii)
    
    if nor == 1:
        return lor
    
    # If lor>1, find the next dominant orientation by locating the
    # next largest count on the bounday that is away from the already detected
    # most dominant orientations by at least sp units (with default value 10 degrees).
    keeptrack = np.ones(m)                  # knowing the location of the detected peak, 
    for l in range(loc[0]-sp, loc[0]+sp):   # avoid testing neighnorhoods of the peak
                                            # for remaining peaks
        if l<0:    
            keeptrack[l+m] = 0
        elif l>=m:
            keeptrack[l-m] = 0
        else:
            keeptrack[l] = 0
            
    for k in range(1,nor):
        t = 0      
        ii = 0
        for i in range(m):
            if bound[i] > maxval[k] and keeptrack[i] == 1:
                maxval[k] = bound[i]
                ii = i
                t = int(i*360.0/m+0.5)
        lor.append(t)
        loc.append(ii)
        
        for l in range(loc[k]-sp, loc[k]+sp):   
            if l<0:    
                keeptrack[l+m] = 0
            elif l>=m:
                keeptrack[l-m] = 0
            else:
                keeptrack[l] = 0

    return lor