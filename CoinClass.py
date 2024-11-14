


from tabnanny import verbose
import numpy as np
import cv2

import TransformIntensities as ti
import SegmentationMethods as sm
import ContourProcessing as cp
import LoadDisplay as ld
import TransformGeometry as tg

# This class provides all the functions needed to read an image of coins,
# extract the coints, create augmented coins from the
# extracted ones and save them for use as training data.
class Coins:
    def __init__(self):
        self.filename = ""  
        self.image = np.array((0,0), dtype=np.uint8)
        self.coins = []          # Detected coins
        self.coinlbls = []       # Labels of detected coins, to be provided by user
        self.augcoins = []       # Augmented detected coins 
        self.augcoinlbls = []    # Labels of augmented coins, obtained from labels of coins
        self.dataset = []        # Dataset created from augmented coins and their labels
        self.contours = []       # Coin boundaries
        self.labels = []         # A list of labels, for internal use
        self.bboxes = []         # Bounding boxes of detected coins
        self.dim = 2             # Number of dimensions of provided image
        self.nr = 0              # Number of rows in provided image
        self.nc = 0              # Number of columns in provided image
        self.sigma = 5.0         # standard deviation of Gaussian smoother
        self.minlength = 600     # minimum contour length to be a coin boundary
        self.bbval = 128         # intensity of bounding boxes for each coin
        self.coindiameter = 129  # diameter of standardized coin (radius=64)
        self.maxcoins = 254      # max number of coins that can be detected in an image
        self.verbose = False     # Work quietly, do not print/show intermediate results
        self.theta = 90          # Default rotational increment when creating augmented data
        self.displaysize = 640*480 # Max size of displayed images
        self.varyingBackg = True   # If image background is homogeneous, set this to False
        self.location = [2, 2]   # Location of first resized/rescaled image on screen
        self.usefolor = 1        # Use color, if available
        
        return
            
    # Provide an image filename to extract coins from it, specify instantiated object
    # parameters, if needed, and return the extracted coins in a list.
    def extractCoins(self, filename, usecolor = 1, sigma = 5, minl = 100, coind = 129, maxc = 254, theta = 90, varyingBackg = True, verbose = False):
        self.filename = filename 
        self.usecolor = usecolor
        self.image = ld.loadImage(self.filename, self.usecolor)
        self.dim = self.image.ndim
        if self.dim == 3:
            (self.nr, self.nc, _) = self.image.shape
        else:
            (self.nr, self.nc) = self.image.shape
        print(f"Image dimensions:{self.nr}x{self.nc}")

        self.sigma = sigma
        self.minlength = minl
        self.coindiameter = coind
        self.maxcoins = maxc
        self.theta = theta
        self.varyingBackg = varyingBackg
        self.verbose = verbose

        sqrtimg = self.image.copy()
          
        if verbose:
            print("To proceed, press any key on keyboard.")
            ld.displayResizedImage("Original image", self.image, self.displaysize, self.location)
                
        # Remove noise in image
        #nimg = cv2.GaussianBlur(self.image,(0,0),sigmaX=1.0,sigmaY=1.0)
        segmented = []
        if self.dim == 3:
            sqrtimg = ti.transformSqrtColors(self.image)
            if verbose:
                ld.displayResizedImage("SQRT intensity transformed", sqrtimg, self.displaysize, self.location)
            # Blur coin details
            img = cv2.GaussianBlur(sqrtimg,(0,0),sigmaX=self.sigma,sigmaY=self.sigma)
            if self.verbose:
                ld.displayResizedImage("Gaussian smoothed", img, self.displaysize, self.location)
            hue, sat, val = ti.convertBGR2HSV(img, False)
            #if self.verbose:
            #    ld.displayResizedImage("Hue component", hue, self.displaysize, self.location)
            #    ld.displayResizedImage("Saturation component", sat, self.displaysize, self.location)
            #    ld.displayResizedImage("Value component", val, self.displaysize, self.location)
            if self.varyingBackg:
                hue = sm.thresholdingBilinearOtsu(hue)
                sat = sm.thresholdingBilinearOtsu(sat)
                val = sm.thresholdingBilinearOtsu(val)
            else:
                hue = sm.thresholdingOtsu(hue)
                sat = sm.thresholdingOtsu(sat)
                val = sm.thresholdingOtsu(val)
            val = sm.notImage(val)
            segmented = sm.orImages(sat,val)
        else:
            sqrtimg = ti.transformSqrtIntensities(self.image)
            if verbose:
                ld.displayResizedImage("SQRT intensity transformed", sqrtimg, self.displaysize, self.location)
            # Blur coin details
            img = cv2.GaussianBlur(sqrtimg,(0,0),sigmaX=self.sigma,sigmaY=self.sigma)
            if self.verbose:
                ld.displayResizedImage("Gaussian smoothed", img, self.displaysize, self.location)

            if self.varyingBackg:
                segmented = sm.thresholdingBileanrOtsu(img)
            else:
                segmented = sm.thresholdingOtsu(img)
        
        # Make sure a region does not fall on a segmented image boundary
        rows,cols = segmented.shape
        if segmented[0,0] == 0 and segmented[rows-1,0] == 0 and \
            segmented[0,cols-1] == 0 and segmented[rows-1,cols-1] == 0:
            backg = 0               # If background is black make sure object (255) does
                                    # not appear on image border
        elif segmented[0,0] == 255 and segmented[rows-1,0] == 255 and \
            segmented[0,cols-1] == 255 and segmented[rows-1,cols-1] == 255:                       
            backg = 255             # If background is while make sure object (0) does
                                    # not appear on image border
            
            for i in range(rows):
                segmented[i,0] = backg
                segmented[i,cols-1] = backg
            for j in range(cols):
                segmented[0,j] = backg
                segmented[rows-1,j] = backg

        if self.verbose:
            ld.displayResizedImage("Segmented image", segmented, self.displaysize, self.location)

        # Find region boundaries
        by = cp.findBoundaries(segmented)
        if self.verbose:
            ld.displayResizedImage("Region boundaries", by, self.displaysize, self.location)
        contours = cp.findContours(by)    
        byy = cp.drawBoundaries(by,contours)  
        if self.verbose:
                ld.displayResizedImage("Detected contours", byy, self.displaysize, self.location)
        if (len(contours)) > 1:
            contours = cp.removeInteriorContours(contours)  
            
        byy = cp.drawExteriorBoundaries(by,contours) 
        if self.verbose:
            ld.displayResizedImage("Exterior contours", byy, self.displaysize, self.location)
        
        # To remove possible shadows around parts of a coin, fit a circle
        # to the contour of coin, and replace the contour with the circle
        contours = cp.getBestFittingCircularContours(contours)
        
        # Remove contours too short to be coin boundaries
        contcopy = contours.copy()
        contours = []
        for i in range(len(contcopy)):
            if len(contcopy[i]) >= self.minlength:
                contours.append(contcopy[i])

        byy = cp.drawExteriorBoundaries(by,contours)
        if self.verbose:
            ld.displayResizedImage("Circle-fitted contours", byy, self.displaysize, self.location)
                
        bboxes = cp.findBoundingBoxes(contours)

        # Before proceeding, check and correct any errors made so far
        n = len(contours)
        ncontours = [] 
        nbboxes = []
        for i in range(n):
            (minr,minc,maxr,maxc) = bboxes[i]
            if minr < maxr and minc < maxc:
                ncontours.append(contours[i])
                nbboxes.append(bboxes[i])
        contours = ncontours
        bboxes = nbboxes

        img = cp.drawBoundingBoxes(byy,bboxes)
        if self.verbose:
            ld.displayResizedImage("Bounding boxes of contours", img, self.displaysize, self.location)

        labels, limg = cp.suppressBackground(byy,bboxes)

        # Remove nonlabeled noisy boundaries
        rows,cols = limg.shape
        for i in range(rows):
            for j in range(cols):
                if limg[i,j]==255:
                    limg[i,j]=0
        if self.verbose:
            ld.displayResizedImage("Background suppressed labeled regions", limg, self.displaysize, self.location)
        
        regions = cp.extractRegions(sqrtimg,limg,bboxes)

        cv2.destroyAllWindows()
        if self.verbose:
            ld.displayScaledImages("Final detected coin regions", regions, 2)

        # Remove contours, labels, bboxes, regions that are not round enough to represent coins
        ncontours = []
        nlabels = [] 
        nbboxes = [] 
        nregions = []
        #if self.verbose:
        #    print("Characteristics of detected coin regions:")
        for i in range(len(contours)):
            perimeter = len(contours[i])
            area = labels[i][1] + perimeter
            circularity= 4.0*3.14156*area/(perimeter*perimeter)
            if circularity > 1.0 and circularity < 1.5:
                ncontours.append(contours[i])
                nlabels.append(labels[i])
                nbboxes.append(bboxes[i])
                nregions.append(regions[i])
                #if verbose:
                #    print(f"{i}:\t perimeter {perimeter},\t area {area}\t circularity {circularity:.4f}"), 
        
        self.contours = ncontours
        self.labels = nlabels
        self.bboxes = nbboxes
        # Resize coins to 129x129
        for i in range(len(nregions)):
            coin = tg.resize(nregions[i], self.coindiameter)
            self.coins.append(coin)
        if self.verbose:
            print(f"{len(self.coins)} coins detected and resized to 129x129.")

        return self.coins

    # Methods associated with Coins class are as follows:
    # Create labels for the detected coins
    def createCoinLabels(self):
        self.coinlbls = []
        coins = []
        n = len(self.coins)
        if n == 0:
            print("coins list is empty.")
            return
        print(f"Enter labels for displayed coins, valid labels are between 0 and 4,")
        print("0 for penny, 1 for nickle, 2 for dime, 3 for quarter, and 4 for dollar.")
        print("If a displayed coin is not a valid coin, enter label -1.")
        print("Close the image window by pressing a key on keyboard; then enter the label.")
        for k in range(n):
            coin = self.coins[k]
            ld.displayScaledImage(f"{k}", coin, 3, self.location)
            cv2.destroyWindow(f"{k}")
            l = input(f"Enter label for coin {k}:")
            if int(l) < 0:
                print("Discarding this coin.")
                continue
            else:
                coins.append(coin)
                self.coinlbls.append(int(l))
        self.coins = coins.copy()

    # Augment detected coins with additional coins obtained by rotaing 
    # each coin about its center from t=0 up to t=360 with increments of theta degrees.
    # Also, let labels for the rotated coins be the label of the coin used  
    # to obtain the rotated coins.
    def createAugmentedLabeledCoins(self, theta = 90):
        self.theta = theta   # If theta is not provided, use 90 deg as the default
        self.augcoins = []
        self.augcoinlbls = []
        for k in range(len(self.coins)):
            t = self.theta
            self.augcoins.append(self.coins[k])
            self.augcoinlbls.append(self.coinlbls[k])
            while t < 360:
                self.augcoins.append(tg.rotateImage(self.coins[k],t))
                self.augcoinlbls.append(self.coinlbls[k])
                t += self.theta
        self.dataset = (self.augcoins, self.augcoinlbls)
        if self.verbose:
            print("Augmented coins, their labels, and a dataset from them were created.")

        return
    
    # Determine up to 5 dominant orientations of each coin
    # and reorient the detected coin to its dominant orientations to be
    # used as augmented data. When recognizing a coin, again find its
    # dominant orientation, reorient the coin to its dominant
    # orientation, and recognize it using the model, which knows its dominant orientations.
    # Let labels of augmented coins be the label of the coin used to create them.  
    def createOrientationCorrectedAugmentedLabeledCoins(self, orientations = 3):
        if orientations < 1 or orientations > 5:
            print("Parameter 'orientations' is out of range, 3 was assumed.")
            orientations = 3
            
        self.augcoins = []
        self.augcoinlbls = []
        
        n = len(self.coins)
        for i in range(n):
            coin = self.coins[i]
            lbl = self.coinlbls[i]
            lor = tg.findDominantOrientations(coin, orientations)
            m = len(lor)
            if m < orientations:
                print(f"Only {m} dominant orientations found.")
            for j in range(m):
                rcoin = tg.rotateImage(coin,lor[j])
                self.augcoins.append(rcoin)
                self.augcoinlbls.append(lbl)
            self.dataset = (self.augcoins, self.augcoinlbls)
        if self.verbose:
            print("Augmented coins, their labels, and dataset were created.")
            
        return

    # Display detected coins
    def displayCoins(self):
        ld.displayScaledImages("Coin",self.coins, 2)

    # Display augmented coins
    def displayAugmentedCoins(self):
        ld.displayImages("augCoin",self.augcoins)
        
    # Display labeled coins
    def displayLabeledCoins(self, coinTypes):
        ld.displayLabeledCoins(self.coinlbls,self.coins, coinTypes)

    # Save detected coins
    def saveCoins(self):
        import os
        import pickle
        basename = os.path.basename(self.filename)  # get basename
        dir = os.path.dirname(self.filename)        # get dir name
        parts = basename.rsplit(".")                # get file format
        name = parts[0]     
        if dir == "":
            dir = name
        else:
            dir = dir+"/"+name
        if not os.path.exists(dir):     # If dir does not exist,
            os.mkdir(dir)               # create it. 
        dirpath = os.path.join(dir,name)
        with open(dirpath, 'wb') as fp:
            pickle.dump(self.coins, fp)
            fp.close()
        if self.verbose: 
            print(f"Saved coins in file {dirpath}.")

    # Save coin labels
    def saveCoinLabels(self):
        import os
        import pickle
        basename = os.path.basename(self.filename)  # get basename
        dir = os.path.dirname(self.filename)        # get dir name
        parts = basename.rsplit(".")                # get file format
        name = parts[0]     
        if dir == "":
            dir = name
        else:
            dir = dir+"/"+name
        if not os.path.exists(dir):     # If dir does not exist
            os.mkdir(dir)               # create it; otherwise, use it  
        dirpath = os.path.join(dir,name+"lbls")
        with open(dirpath, 'wb') as fp:
            pickle.dump(self.coinlbls, fp)
            fp.close()
        if self.verbose:
            print(f"Saved coin labels in file {dirpath}.")

    # Save augmented coins
    def saveAugCoins(self):
        import os
        import pickle
        basename = os.path.basename(self.filename)  # get basename
        dir = os.path.dirname(self.filename)        # get dir name
        parts = basename.rsplit(".")                # get file format
        name = parts[0]     
        if dir == "":
            dir = name
        else:
            dir = dir+"/"+name
        if not os.path.exists(dir):     # If dir does not exist,
            os.mkdir(dir)               # create it.  
        dirpath = os.path.join(dir,name+"aug")
        
        with open(dirpath, 'wb') as fp:
            pickle.dump(self.augcoins, fp)
            fp.close()
        if self.verbose: 
            print(f"Saved augcoins in file {dirpath}.")

    # Save augmented coin labels
    def saveAugCoinLabels(self):
        import os
        import pickle
        basename = os.path.basename(self.filename)  # get basename
        dir = os.path.dirname(self.filename)        # get dir name
        parts = basename.rsplit(".")                # get file format
        name = parts[0]     
        if dir == "":
            dir = name
        else:
            dir = dir+"/"+name
        if not os.path.exists(dir):     # If dir does not exist,
            os.mkdir(dir)               # create it.  
        dirpath = os.path.join(dir,name+"auglbls")
        
        with open(dirpath, 'wb') as fp:
            pickle.dump(self.augcoinlbls, fp)
            fp.close()
        if self.verbose:
            print(f"Saved augcoin labels in file {dirpath}.")

        return
    
    # Create a dataset from augcoins and augcoinlbls
    def createDataset(self):
        if len(self.augcoins) == 0:
            print("augcoins not found, first create it using detected coins.")
            return
        if len(self.augcoinlbls) == 0:
            print("augcoinlbls not found, first create it using coinlbls.")
            return
        self.dataset = (self.augcoins, self.augcoinlbls)
        
        return


    # Save dataset, which consists of augcoins and augcoinlbls, to default file
    def saveDataset(self):
        import os
        import pickle
        if len(self.dataset) ==0:
            print("No dataset found, first create it using augcoins and augcoinlbls.")
            return
        
        basename = os.path.basename(self.filename)  # get basename
        dir = os.path.dirname(self.filename)        # get dir name
        parts = basename.rsplit(".")                # get file format
        name = parts[0]     
        if dir == "":
            dir = name
        else:
            dir = dir+"/"+name
        if not os.path.exists(dir):     # If dir does not exist,
            os.mkdir(dir)               # create it.  
        dirpath = os.path.join(dir,name+"dataset")
        
        with open(dirpath, 'wb') as fp:
            pickle.dump(self.dataset, fp)
            fp.close()
        if self.verbose:
            print(f"Saved a dataset of {len(self.augcoins)} augmented coins in file {dirpath}.")

        return
    
    # Save dataset to the provided filename
    def saveDatasetTo(self, filename):
        import pickle
        if len(self.dataset) ==0:
            print("No dataset found, first create it using augcoins and augcoinlbls.")
            return
                
        with open(filename, 'wb') as fp:
            pickle.dump(self.dataset, fp)
            fp.close
        if self.verbose:
            print(f"Saved a dataset of {len(self.augcoins)} labeled augmented coins in file {filename}.")

        return

    # Save labeled coins to the provided filename
    def saveLabeledCoinsTo(self, filename):
        import pickle
        if len(self.coins) == 0:
            print("No coins found, first extract coins from an image.")
            return
        if len(self.coinlbls) == 0:
            print("Coin labels are not found, first label the coins.")
            return
                
        labeledCoins = (self.coins,self.coinlbls)
        with open(filename, 'wb') as fp:
            pickle.dump(labeledCoins, fp)
            fp.close
        if self.verbose:
            print(f"Saved a labeled-coin dataset of {len(self.coins)} coins in file {filename}.")

        return
    
    # Load coins from specified file
    def loadCoins(self, filename):
        import pickle
        with open(filename, 'rb') as fp:
            self.coins = pickle.load(fp)
        if self.verbose:
            print("Loaded coins.")
        if self.verbose:
            print(f"Loaded {len(self.coins)} coins.")

    # Load coin labels from specified file
    def loadCoinLabels(self,filename):
        import pickle
        with open(filename, 'rb') as fp:
            self.coinlbls = pickle.load(fp)
        if self.verbose:
            print("Loaded coin labels.")
        if self.verbose:
            print(f"Loaded {len(self.coinlbls)} coin labels.")

    # Load a dataset containing augcoins and augcoinlbls from provided file
    def loadDataset(self,filename):
        import pickle
        with open(filename, 'rb') as fp:
            self.dataset = pickle.load(fp)
            (self.augcoins, self.augcoinlbls) = self.dataset
        if self.verbose:
            print(f"Loaded a dataset containing {len(self.augcoins)} labeled coins.")

        return self.dataset

    # Load a labeled-coins dataset from filename
    def loadLabeledCoins(self,filename):
        import pickle
        labeledCoins = []
        with open(filename, 'rb') as fp:
            labeledCoins = pickle.load(fp)
            (self.coins, self.coinlbls) = labeledCoins
            # Consider the labeled coins are augmented labeled coins also
            (self.augcoins, self.augcoinlbls) = labeledCoins
        if self.verbose:
            print(f"Uploaded a dataset cointaining {len(self.coins)} labeled coins.")

        return self.dataset
    
    # Get self.dataset
    def getDataset(self):
        return self.dataset

    # Get self.coins
    def getCoins(self):
        return self.coins

    # Get self.augcoins
    def getAugCoins(self):
        return self.augcoins

    # Save detected coins to default file
    def saveDetectedCoins(self):
        import os
        n = len(self.coins)
        if n == 0:
            print("No coins detected!")
        else:
            basename = os.path.basename(self.filename) # get basename
            dir = os.path.dirname(self.filename)       # get dir name
            parts = basename.rsplit(".")               # get file format
            name = parts[0]   
            fmt = "png" #parts[1]   
            if dir == "":
                dir = name
            else:
                dir = dir+"/"+name
            if not os.path.exists(dir):     # If dir does not exist,
                os.mkdir(dir)               # create it.  
            for i in range(n):
                nname = name+f"{i}."+fmt    # Add to base name image number, a dot, and format
                dirpath = os.path.join(dir,nname)   # Concatenate dir and nname
                cv2.imwrite(dirpath,self.coins[i])      # Save image i at dirpath
            if verbose: 
                print(f"{n} coins saved in directory {dir}.")

    # Save created augmented coins to default file
    def saveCreatedAugCoins(self):
        import os
        n = len(self.augcoins)
        if n == 0:
            print("No augmented coins found!")
        else:
            basename = os.path.basename(self.filename) # get basename
            dir = os.path.dirname(self.filename)       # get dir name
            parts = basename.rsplit(".")               # get file format
            name = parts[0]   
            fmt = "png" #parts[1]   
            if dir == "":
                dir = name+"aug"
            else:
                dir = dir+"aug/"+name
            if not os.path.exists(dir):     # If the dir does not exist,
                os.mkdir(dir)               # create it; otherwise, use it  
            for i in range(n):
                nname = name+f"{i}."+fmt    # Add to base name image number, a dot, and its format
                dirpath = os.path.join(dir,nname)   # Concatenate dir and nname
                cv2.imwrite(dirpath,self.augcoins[i])      # Save ith augmented coin
            if verbose: 
                print(f"{n} augmented coins saved in directory {dir}.")
    
    # Replace self.dataset with concatenation of datasets in specified files. 
    def concatenateTwoDatasets(self, file1, file2):
        self.loadDataset(file2)
        (augs,auglbls) = self.dataset    # Contents of file2
        self.loadDataset(file1)     # Move contents of file1 to self.dataset
        for k in range(len(augs)):  # Append contents of file2 to self.dataset
            self.augcoins.append(augs[k])
            self.augcoinlbls.append(auglbls[k])

        self.dataset = (self.augcoins, self.augcoinlbls)

    # Concatenate all datasets in filesdir and save the result in allfiles
    def concatenateAllDatasets(self,filesDir,allfiles):
        import os
        import pickle
 
        # Get list of all files in filesDir
        files = os.listdir(filesDir) 
        n = len(files)
        if n == 0:
            print(f"{filesDir} is empty.")
            return
        elif n == 1:
            filename = f"{filesDir}/{files[0]}"
            self.loadDataset(filename)
            with open(allfiles, 'wb') as fp:
                pickle.dump(self.dataset, fp)
                fp.close()
            if verbose:
                print("Saved the single dataset in file:",allfiles)
        else:
            filename = f"{filesDir}/{files[0]}"
            self.loadDataset(filename)
            (augs,auglbls) = self.dataset
            for i in range(1,n):
                filename = f"{filesDir}/{files[i]}"
                self.loadDataset(filename)
                for k in range(len(self.augcoins)):  
                    augs.append(self.augcoins[k])
                    auglbls.append(self.augcoinlbls[k])
            dataset = (augs, auglbls)
            self.shuffleDataset()      # Shuffle the merged dataset
            with open(allfiles, 'wb') as fp:
                pickle.dump(dataset, fp)
                fp.close()
            print(f"Merged datasets in dir {filesDir}, shuffled, and saved in file: {allfiles}")

    # Shuffle self.dataset
    def shuffleDataset(self):
        import random
        (augs, auglbls) = self.dataset
        n = len(augs)
        if n == 0:
            print("dataset is empty!")
        else:
            self.augcoins = []
            self.augcoinlbls = []
            i = 0
            while i < n:
                m = len(augs)-1
                j = random.randint(0,m)
                aug = augs[j]           # Get the jth element, then
                del augs[j:j+1]         # remove the jth element
                self.augcoins.append(aug)
                lbl = auglbls[j]
                del auglbls[j:j+1]
                self.augcoinlbls.append(lbl)
                i += 1

            self.dataset = (self.augcoins, self.augcoinlbls)


