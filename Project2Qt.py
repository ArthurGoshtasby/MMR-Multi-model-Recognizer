
# This module builds the interface to the multi-model recognizer. The 
# interface uses a tabbed view to show different steps in the creation and 
# testing of the recognizer. Each view is defined by a combination of linear
# horizontal and vertical layouts. The tabbed view is shown at the top of the
# interface by a sequence of horizontal push buttons. Selecting a tab will 
# display an array of push buttons within a stacked layout. By pressing a 
# button within a stacked layout, a part of the work to create a dataset, create 
# a model, test a model, or use a combination of the created models to recognize 
# a class of objects will be carried out. As an example of objects, US coins 
# will be used. Recognition of other objects requires replacing the coin segmentation
# method with a segmentation method that can detect the required object types in an image
# for training and recognition.

from turtle import window_width
import CoinClass as cc
import RecognizeCoin as rc
import LoadDisplay as ld
import vgg16Recognizer as vr
import mobilenetRecognizer as mr
import cv2
import tensorflow as tf
import sys

# PyQt6 is used to design the user interfaces. For more information about PyQt6, 
# visit https://www.pythonguis.com/pyqt6-tutorial/
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *

# US coin types: 0: Penny, 1: Nickle, 2: Dime, 3: Quarter, 4: Dollar
coinTypes = ["Penny", "Nickle", "Dime", "Quarter", "Dollar"]
# Coin values in cents
coinValues = [1, 5, 10, 25, 100]

# The following global parameters are shared among different windows
datafn = ""     # A user specified dataset filename
modelid = ""    # A user selected model type
modelfn = ""    # A user specified model filename
modelsdir = ""  # A user specified directory containing various models
imgfn = ""      # A user specified image filename
model1 = None   # A trained model

# This window acts as the interface for the user to provide the filename of an image
# to be segmented, a filename to saved the detected coins along with their types/labels,
# and a filename to save a training dataset created from the detected coins and a
# selected number of orientations the coins may appear in a previously not seen image.
class Window1(QWidget): 
    
    def __init__(self):
        super().__init__()

        self.imgfn = ""
        self.datafn = ""
        self.ofilename = ""
        self.afilename = ""
        self.verbose = False
        global datafn

        widget0 = QLabel("Provide Filenames of Image, \nLabeled Coins, and Training Dataset")
        font0 = widget0.font()
        font0.setPointSize(20)
        widget0.setFont(font0)

        widget1 = QLineEdit()
        widget1.setMaxLength(80)
        widget1.setPlaceholderText("Image Filename")
        widget1.returnPressed.connect(self.return_pressed1)
        widget1.textEdited.connect(self.text_edited1)

        widget2 = QLineEdit()
        widget2.setMaxLength(80)
        widget2.setPlaceholderText("Filename to Save Labeled Coins")
        widget2.returnPressed.connect(self.return_pressed2)
        widget2.textEdited.connect(self.text_edited2)

        widget3 = QLineEdit()
        widget3.setMaxLength(80)
        widget3.setPlaceholderText("Filename to Save Training Dataset")
        widget3.returnPressed.connect(self.return_pressed3)
        widget3.textEdited.connect(self.text_edited3)

        widget4 = QCheckBox()
        widget4.stateChanged.connect(self.show_state)

        btn = QPushButton("Return")
        btn.setFont(QFont('Arial', 15)) 
        # changing color of button 
        #btn.setStyleSheet("background-color: yellow; color: magenta;")
        btn.pressed.connect(self.return1)

        widget5 = QLabel("Show Intermediate Results")
        font = widget4.font()
        font.setPointSize(15)
        widget5.setFont(font)
        widget5.setAlignment(Qt.AlignmentFlag.AlignLeft)

        layout1 = QHBoxLayout()
        layout1.addWidget(widget4)
        layout1.addWidget(widget5)
        layout1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        widget6 = QWidget()
        widget6.setLayout(layout1)

        layout2 = QVBoxLayout()
        layout2.addWidget(widget0)
        layout2.addWidget(widget6)
        layout2.addWidget(widget1)
        layout2.addWidget(widget2)
        layout2.addWidget(widget3)
        layout2.addWidget(btn)

        self.setLayout(layout2)

    def return1(self):
        if self.imgfn == "" or self.ofilename == "" or self.afilename == "":
            print("Provide all three filenames.")
            return
        print("Image Filename:", self.imgfn, "\nLabeled Coins Filename:",
              self.ofilename, "\nTraining Dataset Filename:",self.afilename,
              "\nVerbose:",self.verbose)
        self.close()
        global window
        window.hide()
        if self.verbose:
            print("Note: After an image is displayed, to proceed,")
            print("make sure the image is active and press any key on keyboard.")

        # Step 1. Preparing a training dataset

        # Step 1.1. Take images of coins 
        # To simplify detection and extraction of coins in images, take
        # images of coins laid within homogeneous backgrounds. To prepare a
        # training dataset, place a dozen or so coins on a homogeneous background
        # such as a white surface. To avoid shadows of coins onto the
        # background, avoid directional lighting and use ambient lighting. Also,
        # to avoid image of a coin region appearing elliptic, take images from
        # above the coins so the coins will appear as much as possible circular.
        # Keep distance of the camera to the coins or set the resolution of the
        # camera so that a coin appears greater than 128x128 but not too much
        # larger than that. Large coin images will simply slow down the 
        # segmentation process without additional benefits. Example images can be 
        # found in directory Images.

        # Step 1.2. Extract individual coins in an image and label the coins.
        # Use method extractCoins() in class Coins() to segment a coin image
        # into individual coins as follows:
        CO = cc.Coins() # Create an object of class Coins() and call it CO.

        # Extract coins in image imgfn.
        CO.extractCoins(self.imgfn, sigma = 5, verbose = self.verbose, varyingBackg = False)
        # sigma is the standard deviation of the Gaussian smoother. This 
        # parameter depends on image size. varyingBackground parameter is set to False
        # if lighting across the image is homogeneous. For a less homogeneous
        # lighting during imaging, set this parameter to True.
        # Set parameter verbose to True to display the intermediate results; otherwise,
        # set this parameter to False. 
        # 
        # To remove displayed images on screen, click on any image to make it active 
        # and then press any key on keyboard.
        cv2.destroyAllWindows()

        # Step 1.3. Interactively provide labels for the extracted coins
        CO.createCoinLabels()   # This method will display each coin and ask the user
                                # to enter the type of the coin (a number between 0 and 4).
        cv2.destroyAllWindows() 

        # Save the coins along with their labeled in self.ofilename.
        # In the following, a dataset of coins along with their labels will be referred to as
        # a Base Dataset.
        CO.saveLabeledCoinsTo(self.ofilename)
        print("A dataset of coins along with their labels (a base dataset) was saved in file:",self.ofilename)

        # Step 1.4. Create augmented coins and their labels, they constitute a training dataset.
        # Any nonsymmetric shape has a dominant orientation. Due to the particular
        # lighting condition of the imaging environment, the most dominant orientation of a
        # coin may be its second or third most dominant orientation. To reorient an extracted
        # coin in the direction of its first 5 most dominant orientations are saved 
        # as augmented coins with their associating labels. Labeled augmented coins will be
        # used as a training dataset to train a model.
        CO.createOrientationCorrectedAugmentedLabeledCoins(orientations = 5)

        if self.verbose:
            # The following will display the augmented/training dataset.
            ld.displayImages("Aug. Coins",CO.augcoins)
            print("Press a key to remove displayed augmented/training dataset.")
            cv2.destroyAllWindows()     # Press any key to remove displayed images 

        # Shuffle the coins together with their labels so that if a dataset is 
        # partitioned into training and validation subsets, information about the coins 
        # appear in both training and validation datasets.
        CO.shuffleDataset()

        # Save the augmented coins along with their labels as a training dataset
        # in self.afilename provided earlier by the user.
        CO.saveDatasetTo(self.afilename)
        print("Shuffled and saved the augmented/training dataset in file:",self.afilename)
        self.datafn = self.afilename
        datafn = self.datafn        # Make this dataset globally visible
        window.show()

    def text_edited1(self, s):
        self.imgfn = s

    def return_pressed1(self):
        if self.imgfn == "":
            return
        else:
            self.return1()

    def text_edited2(self, s):
        self.ofilename = s

    def return_pressed2(self):
        if self.ofilename == "":
            return
        else:
            self.return1()

    def text_edited3(self, s):
        self.afilename = s

    def return_pressed3(self):
        if self.afilename == "":
            return
        else:
            self.return1()

    def show_state(self, s):
        if s == 2:
            self.verbose = True

# This window provides the interface for the user to specify the directory containing
# two or more training datasets and a filename to save the merged datasets. 
class Window2(QWidget): 
    
    def __init__(self):
        super().__init__()

        self.dirn = ""
        self.mergedfn = ""

        widget0 = QLabel("Merge Datasets in a Directory")
        font0 = widget0.font()
        font0.setPointSize(20)
        widget0.setFont(font0)

        widget1 = QLineEdit()
        widget1.setMaxLength(80)
        widget1.setPlaceholderText("Directory Containing Datasets")
        widget1.returnPressed.connect(self.return_pressed)
        widget1.textEdited.connect(self.text_edited1)

        widget2 = QLineEdit()
        widget2.setMaxLength(80)
        widget2.setPlaceholderText("Filename to Save Merged Datasets")
        widget2.returnPressed.connect(self.return_pressed)
        widget2.textEdited.connect(self.text_edited2)

        btn = QPushButton("Return")
        btn.setFont(QFont('Arial', 15)) 
        btn.pressed.connect(self.return2)

        layout = QVBoxLayout()
        layout.addWidget(widget0)
        layout.addWidget(widget1)
        layout.addWidget(widget2)
        layout.addWidget(btn)

        self.setLayout(layout)

    def return2(self):
        if self.dirn == "" or self.mergedfn == "":
            print("Provide both directory of datasets and filename to save merged datasets.")
            return
        self.close()
        global window
        window.hide()
        # To merge all datasets in self.dirn and save the merged datasets
        # in file self.mergedfn, use:
        CO = cc.Coins()
        CO.concatenateAllDatasets(self.dirn,self.mergedfn)
        window.show()

    def return_pressed(self):
        if self.dirn == "" or self.mergedfn == "":
            return
        else:
            self.return2()

    def text_edited1(self, s):
        self.dirn = s

    def text_edited2(self, s):
        self.mergedfn = s

# This window provides the interface for the user to specify the filename of a training
# dataset, to select the type of a model to be created, and specify the filename to save 
# the mode after training.
class Window3(QWidget): 
    
    def __init__(self):
        super().__init__()

        self.datafn = ""
        self.modelid = ""
        self.modelfn = ""

        widget0 = QLabel("Provide a Training Dataset, a Model Type, and a Model Filename\n")
        font0 = widget0.font()
        font0.setPointSize(20)
        widget0.setFont(font0)

        widget1 = QLineEdit()
        widget1.setMaxLength(80)
        widget1.setPlaceholderText("Filename of Training Dataset")
        widget1.returnPressed.connect(self.return_pressed1)
        widget1.textEdited.connect(self.text_edited1)

        widget2 = QLabel("\nSelect One of the Following Model Types")
        font0 = widget0.font()
        font0.setPointSize(18)
        widget2.setFont(font0)

        widget3 = QListWidget()
        widget3.addItems(["Fully Connected NN", "Convolutional NN", 
                         "Fine-tuned Mobilenet", "Fine-tuned VGG16"])
        font1 = widget0.font()
        font1.setPointSize(15)
        widget3.setFont(font1)
        widget3.currentTextChanged.connect(self.text_changed)

        widget4 = QLineEdit()
        widget4.setMaxLength(80)
        widget4.setPlaceholderText("Filename to Save Trained Model")
        widget4.returnPressed.connect(self.return_pressed2)
        widget4.textEdited.connect(self.text_edited2)

        btn = QPushButton("Return")
        btn.setFont(QFont('Arial', 15)) 
        btn.pressed.connect(self.return3)

        layout = QVBoxLayout()
        layout.addWidget(widget0)
        layout.addWidget(widget1)
        layout.addWidget(widget2)
        layout.addWidget(widget3)
        layout.addWidget(widget4)
        layout.addWidget(btn)

        self.setLayout(layout)

    def return3(self):
        if self.datafn == "" or self.modelid == "" or self.modelfn == "":
            print("Provide a dataset, a model type, and a model filename.")
            return
        self.close()
        global datafn, modelid, modelfn
        datafn = self.datafn
        modelid = self.modelid
        if modelid == "Fully Connected NN":
            modelfn = self.modelfn + "FCNN.h5"
        elif modelid == "Convolutional NN":
            modelfn = self.modelfn + "CNN.h5"
        elif modelid == "Fine-tuned Mobilenet":
            modelfn = self.modelfn + "FTM.h5"
        elif modelid == "Fine-tuned VGG16":
            modelfn = self.modelfn + "FTVGG.h5"
        
        print("Dataset Filename:",datafn, "\nModel Type:",
              modelid, "\nModel Filename:",modelfn)
        print("Note: Model filename was extended to include model-type identifier.")

        # Step 2. Training a neural network to recognize coins.
        # Design a network from scratch or fine tune a previously designed one.
 
        from keras import models
        global model1
        model1 = models.Sequential()
        if modelid == "Fully Connected NN":
            # Here a fully connected neural network is designed from scratch.
            model1 = rc.BasicRecognizer(datafn, modelfn, coinTypes, epochs = 50)
        elif modelid == "Convolutional NN":
            # Here a convolutional neural network is designed from scratch.
            model1 = rc.CNNRecognizer(datafn, modelfn, coinTypes, epochs = 20)
        elif modelid == "Fine-tuned Mobilenet":
            # Here the well-known Mobilenet network is fine tuned.
            model1 = mr.mobilenetTrainModel(datafn, modelfn, coinTypes, epochs = 10)
        elif modelid == "Fine-tuned VGG16":
            # Here the well-known VGG16 network is fine tuned.
            weights_path='VGG16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            model1 = vr.vgg16TrainModel(datafn, modelfn, weights_path, coinTypes, epochs = 20)

    def return_pressed1(self):
        if self.datafn == "":
            return
        else:
            self.return3()

    def text_edited1(self, fn):   
        self.datafn = fn

    def return_pressed2(self):
        if self.modelfn == "":
            return
        else:
            self.return3()

    def text_edited2(self, fn):   
        self.modelfn = fn

    def text_changed(self, id): 
        self.modelid = id
        self.return3() 
        
# This window provides the interface for the user to specify the filename of
# a previously trained model and a test dataset filename to be used to evaluate
# the accuracy of the model recognizing coins in the dataset.
class Window4(QWidget): 
    
    def __init__(self):
        super().__init__()

        self.modelfn = ""
        self.datafn = ""
        global model1

        if model1 == None:
            widget0 = QLabel("Provide Model and Dataset Filenames")
            font0 = widget0.font()
            font0.setPointSize(20)
            widget0.setFont(font0)

            widget1 = QLineEdit()
            widget1.setMaxLength(80)
            widget1.setPlaceholderText("Model Filename")
            widget1.returnPressed.connect(self.return_pressed1)
            widget1.textEdited.connect(self.text_changed1)

            widget2 = QLineEdit()
            widget2.setMaxLength(80)
            widget2.setPlaceholderText("Dataset Filename")
            widget2.returnPressed.connect(self.return_pressed2)
            widget2.textEdited.connect(self.text_changed2)

            btn = QPushButton("Return")
            btn.setFont(QFont('Arial', 15)) 
            btn.pressed.connect(self.return4)

            layout = QVBoxLayout()
            layout.addWidget(widget0)
            layout.addWidget(widget1)
            layout.addWidget(widget2)
            layout.addWidget(btn)

            self.setLayout(layout)
        else:
            widget0 = QLabel("Provide a Dataset Filename")
            font0 = widget0.font()
            font0.setPointSize(20)
            widget0.setFont(font0)

            widget1 = QLineEdit()
            widget1.setMaxLength(80)
            widget1.setPlaceholderText("Dataset Filename")
            widget1.returnPressed.connect(self.return_pressed3)
            widget1.textEdited.connect(self.text_edited3)

            btn = QPushButton("Return")
            btn.setFont(QFont('Arial', 15)) 
            btn.pressed.connect(self.return5)

            layout = QVBoxLayout()
            layout.addWidget(widget0)
            layout.addWidget(widget1)
            layout.addWidget(btn)

            self.setLayout(layout)

    def return4(self):
        if self.modelfn == "" or self.datafn == "":
            print("Provide both dataset filename and model filename.")
            return
        self.close()
        global modelfn, datafn
        modelfn = self.modelfn
        datafn = self.datafn
        
        print("Using dataset", datafn, "and model",modelfn)

        # Step 3. Evaluate the recognition accuracy of a created model.
        # Using a test dataset from among base datasets created earlier to evaluate
        # the recognition accuracy of the model.
        n = len(modelfn)
        if modelfn[n-4] == 'G' and modelfn[n-5] == 'G' and modelfn[n-6] == 'V':
            vr.vgg16TestModel(datafn, modelfn, coinTypes)
        else:
            rc.testModel(datafn, modelfn, coinTypes)

    def return_pressed1(self):
        if self.modelfn == "":
            return
        else:
            print("model filename",self.modelfn)
            self.return4()

    def text_changed1(self, fn):   
        self.modelfn = fn

    def return_pressed2(self):
        if self.datafn == "":
            return
        else:
            print("dataset filename", self.datafn)
            self.return4()

    def text_changed2(self, fn):   
        self.datafn = fn

    def return5(self):
        if self.datafn == "":
            print("Provide a dataset filename.")
            return
        self.close()
        global datafn, modelfn, model1
        datafn = self.datafn
        print("Using dataset",datafn, "and the model just trained.")
        n = len(modelfn)
        if modelfn[n-4] == 'G' and modelfn[n-5] == 'G' and modelfn[n-6] == 'V':
            vr.vgg16TestThisModel(datafn, model1, coinTypes)
        else:
            rc.testThisModel(datafn, model1, coinTypes)

    def return_pressed3(self):
        if self.datafn == "":
            return
        else:
            self.return5()

    def text_edited3(self, fn):   
        self.datafn = fn

# This window provides the means for the user to specify the name of the directory
# containing two or more models and the filename of a test (base) dataset.
class Window5(QWidget): 
    
    def __init__(self):
        super().__init__()

        self.modelsdir = ""
        self.datafn = ""

        widget0 = QLabel("Provide the Directory of Models and a Dataset Filename")
        font0 = widget0.font()
        font0.setPointSize(20)
        widget0.setFont(font0)

        widget1 = QLineEdit()
        widget1.setMaxLength(80)
        widget1.setPlaceholderText("Models Directory")
        widget1.returnPressed.connect(self.return_pressed1)
        widget1.textEdited.connect(self.text_changed1)

        widget2 = QLineEdit()
        widget2.setMaxLength(80)
        widget2.setPlaceholderText("Dataset Filename")
        widget2.returnPressed.connect(self.return_pressed2)
        widget2.textEdited.connect(self.text_changed2)

        btn = QPushButton("Return")
        btn.setFont(QFont('Arial', 15)) 
        btn.pressed.connect(self.return4)

        layout = QVBoxLayout()
        layout.addWidget(widget0)
        layout.addWidget(widget1)
        layout.addWidget(widget2)
        layout.addWidget(btn)

        self.setLayout(layout)

    def return4(self):
        if self.modelsdir == "" or self.datafn == "":
            print("Provide both models directory and dataset filename.")
            return
        self.close()
        global modelsdir, datafn
        modelsdir = self.modelsdir
        datafn = self.datafn

        # Step 4. Using multiple models to recognize coins in a dataset or in an image.
        
        # The created models predict coins mostly correctly. However, 
        # they sometime make mistakes and the mistakes are not always at the same for
        # different models. Using a voting process and taking into consideration the 
        # accuracy of each recognizer, we let each model vote for the type of a coin with 
        # the accuracy of the recognizer that has been determined earlier. Then, the coin
        # type receiving the highest vote will be chosen as the most likely type 
        # for the coin. Although this does not guarantee correct prediction of 
        # all coins, it will very likely produce fewer incorrect predictions 
        # than those obtained by any single model.
        
        print("Using dataset", datafn, "and models in directory",modelsdir)

        print("Using the following models:")
        import os
        files = []
        for file in os.listdir(modelsdir):
            files.append(file)
            print(file)

        # Recognize coins in the datasetfn using models in directory modelsdir
        rc.recognizeUsingMultipleModelsNew(modelsdir, coinTypes, coinValues, datasetfn=datafn)

    def return_pressed1(self):
        if self.modelsdir == "":
            return
        else:
            print("Models Directory:",self.modelsdir)
            self.return4()

    def text_changed1(self, fn):   
        self.modelsdir = fn

    def return_pressed2(self):
        if self.datafn == "":
            return
        else:
            print("Dataset Filename", self.datafn)
            self.return4()

    def text_changed2(self, fn):   
        self.datafn = fn

# This window provides the means for the user to specify the name of the directory
# containing two or more models and the filename of an image containing coins. The
# models will then be used to recognize coins in the image.
class Window6(QWidget): 
    
    def __init__(self):
        super().__init__()

        self.modelsdir = ""
        self.imgfn = ""

        widget0 = QLabel("Provide a Directory of Models and an Image Filename")
        font0 = widget0.font()
        font0.setPointSize(20)
        widget0.setFont(font0)

        widget1 = QLineEdit()
        widget1.setMaxLength(80)
        widget1.setPlaceholderText("Models Directory")
        widget1.returnPressed.connect(self.return_pressed1)
        widget1.textEdited.connect(self.text_changed1)

        widget2 = QLineEdit()
        widget2.setMaxLength(80)
        widget2.setPlaceholderText("Image Filename")
        widget2.returnPressed.connect(self.return_pressed2)
        widget2.textEdited.connect(self.text_changed2)

        btn = QPushButton("Return")
        btn.setFont(QFont('Arial', 15)) 
        btn.pressed.connect(self.return4)

        layout = QVBoxLayout()
        layout.addWidget(widget0)
        layout.addWidget(widget1)
        layout.addWidget(widget2)
        layout.addWidget(btn)

        self.setLayout(layout)

    def return4(self):
        if self.modelsdir == "" or self.imgfn == "":
            print("Provide both models directory and image filename.")
            return
        self.close()
        global modelsdir, imgfn
        modelsdir = self.modelsdir
        imgfn = self.imgfn
        
        # We visually evaluate the accuracy of a multi-model
        # recognizer using coins in the provided image.
        print("Using image", imgfn, "and models in directory",modelsdir)

        print("Models in provided directory:")
        import os
        files = []
        for file in os.listdir(modelsdir):
            files.append(file)
            print(file)

        # Visually evaluate the predicted type of a coin in imagefn by models in modelsdir
        # This method first detects coins in the image and then predicts
        # the type of each coin.
        rc.recognizeUsingMultipleModelsNew(modelsdir, coinTypes, coinValues, imagefn=imgfn)

    def return_pressed1(self):
        if self.modelsdir == "":
            return
        else:
            print("Models Directory:",self.modelsdir)
            self.return4()

    def text_changed1(self, fn):   
        self.modelsdir = fn

    def return_pressed2(self):
        if self.imgfn == "":
            return
        else:
            print("Image Filename", self.imgfn)
            self.return4()

    def text_changed2(self, fn):   
        self.imgfn = fn

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-model Recognizer")

        # The structure of the stacked layout 1
        layout1 = QVBoxLayout()
        layout1.setContentsMargins(2,2,2,2)
        layout1.setSpacing(5)

        # Items within the stacked layout associated with tab 1.
        btn1 = QPushButton("Prepare a Dataset")
        btn1.setFont(QFont('Arial', 20)) 
        # changing color of button 
        btn1.setStyleSheet("background-color: yellow; color: magenta;")
        btn1.pressed.connect(self.task1)
        btn2 = QPushButton("Combine Datasets")
        btn2.setFont(QFont('Arial', 20))
        btn2.setStyleSheet("background-color: yellow; color: magenta;")
        btn2.pressed.connect(self.task2)

        layout1.addWidget(btn1) 
        layout1.addWidget(btn2)

        widget1 = QWidget()
        widget1.setLayout(layout1)

        pagelayout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.stacklayout = QStackedLayout()

        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(self.stacklayout)

        btn = QPushButton("Prepare Training Data")
        btn.setFont(QFont('Arial', 12))
        btn.setStyleSheet("background-color: cyan;")
        btn.pressed.connect(self.activate_tab_1)
        button_layout.addWidget(btn)
        self.stacklayout.addWidget(widget1)
        
        # The structure of the stacked layout 2
        layout4 = QVBoxLayout()
        layout4.setContentsMargins(2,2,2,2)
        layout4.setSpacing(5)
        layout5 = QHBoxLayout()
        layout5.setContentsMargins(5,5,5,5)
        layout5.setSpacing(5)
        layout6 = QHBoxLayout()
        layout6.setContentsMargins(5,5,5,5)
        layout6.setSpacing(5)

        # The components of the stacked layout 2
        btn7 = QPushButton("Train Model")
        btn7.setFont(QFont('Arial', 20)) 
        btn7.setStyleSheet("background-color: yellow; color: magenta;")
        btn7.pressed.connect(self.task7)
        btn8 = QPushButton("Test Model")
        btn8.setFont(QFont('Arial', 20))
        btn8.setStyleSheet("background-color: yellow; color: magenta;")
        btn8.pressed.connect(self.task8)

        layout5.addWidget(btn7) 
        layout6.addWidget(btn8)

        layout4.addLayout(layout5)
        layout4.addLayout(layout6)

        widget2 = QWidget()
        widget2.setLayout(layout4)

        btn = QPushButton("Train and Test a Model")
        btn.setFont(QFont('Arial', 12))
        btn.setStyleSheet("background-color: cyan;")
        btn.pressed.connect(self.activate_tab_2)
        button_layout.addWidget(btn)
        self.stacklayout.addWidget(widget2)

        # Structure of stacked layout 3
        layout10 = QVBoxLayout()
        layout10.setContentsMargins(2,2,2,2)
        layout10.setSpacing(5)

        # Items included within stacked layout 3
        btn14 = QPushButton("Recognize Coins in a Dataset")
        btn14.setFont(QFont('Arial', 18)) 
        btn14.setStyleSheet("background-color: yellow; color: magenta;")
        btn14.pressed.connect(self.task14)
        btn15 = QPushButton("Find Total Value of Coins in an Image")
        btn15.setFont(QFont('Arial', 18))
        btn15.setStyleSheet("background-color: yellow; color: magenta;")
        btn15.pressed.connect(self.task15)

        layout10.addWidget(btn14) 
        layout10.addWidget(btn15)

        widget4 = QWidget()
        widget4.setLayout(layout10)

        btn = QPushButton("Test a Multi-model Recognizer")
        btn.setFont(QFont('Arial', 12))
        btn.setStyleSheet("background-color: cyan;")
        btn.pressed.connect(self.activate_tab_3)
        button_layout.addWidget(btn)
        self.stacklayout.addWidget(widget4)

        btn = QPushButton("Exit")
        btn.setFont(QFont('Arial', 12))
        btn.setStyleSheet("background-color: cyan;")
        btn.pressed.connect(self.exitProg)
        button_layout.addWidget(btn)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

    # If tab 1 is selected, show stacked layout 1
    def activate_tab_1(self):
        self.stacklayout.setCurrentIndex(0)

    # If tab 2 is selected, show stacked layout 2
    def activate_tab_2(self):
        self.stacklayout.setCurrentIndex(1)

    # If tab 3 is selected, show stacked layout 3
    def activate_tab_3(self):
        self.stacklayout.setCurrentIndex(2)

    # If tab 4 is selected, exit program
    def exitProg(self):
        print("So long!")
        exit(0)

    # Each of the following functions identify the task to be performed when 
    # a button within the stacked layouts is pressed.
    def task1(self):
        self.w = Window1()
        self.w.show()

    def task2(self):
        self.w = Window2()
        self.w.show()

    def task7(self):
        self.w = Window3()
        self.w.show()

    def task8(self):
        self.w = Window4()
        self.w.show()

    def task14(self):
        self.w = Window5()
        self.w.show()

    def task15(self):
        self.w = Window6()
        self.w.show()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()

'''
Limitations:
This software has the following limitations:
1)  The coin detector and recognizer are limited to circular coins that do not contain 
    holes.
2)  Imaging direction is limited to nadir-view. Side-view images that produce elliptic
    coin boundaries will be ignored and not detected.
3)  The coins should not touch each other or the image borders. Adjacent
    coins should be several pixels from each other.
4)  Background of captured coin images should be homogeneous in intensity. Although
    a slowly varying background intensity may be okay, but sharply varying background
    intensity or use of a textured background will fail the detection process.
    
Usage:
To use the software as is, try to take images that contain fewer than 20 coins, taken under
ambient lighting, and with a homogeneous background. Images of size 1024x1024 or 800x1200
are appropriate. Larger images are okay, but they will require a longer time to process. 
Much smaller images may produce coins that do not contain sufficient 
information to enable recognition. If detected coins are smaller than 129x129, it means
either resolution of the images is not sufficient, or there are too many coins in
the image. 

Upgrades:
This coin detector and recognizer can be upgraded to remove/reduce the above limitations.
1)  To allow coins with a circular hole, after segmentation, rather than removing all 
    boundaries inside a boundary region, the software can be extended to remove all  
    interior boundaries except the one which has about the same center coordinates as the 
    exterior boundary.
2)  To extend the detector to process elliptic coin regions, rather than a circle detector,
    an ellipse detector should be used. This ultimately will require mapping each
    detected elliptic region to a circular one so that different views of the same coin  
    can produce the same circular coin.
3)  To allow coins to touch each other in an image, rather than the segmentation method
    described here, a different segmentation method that does not involve image smoothing
    should be used. Image smoothing merges adjacent coin regions, making detection of 
    individual coins difficult, if not impossible.
4)  To make it possible to use nonhomogeneous backgrounds when capturing coin images,
    a means to identify the background in one piece should be developed so that 
    non-background regions can be assigned to coin regions. 
5)  To extend the capabilities of the segmentation method to include images obtained from
    different view orientations and under different lighting conditions, there is a need to 
    include training images that contain various lighting conditions obtained under 
    different camera orientations. Since directional lighting produces shadows, the  
    segmentation method should be able to separate a coin from its shadow. If the shadow is 
    left in an extracted coin, the segmentation should still be able to extract a coin and its 
    shadow together for training. Allowing directional lighting and arbitrary camera view
    orientation requires a much larger training dataset to cover all possible environmental 
    lighting conditions and camera orientations. In such a situation, perhaps instead 
    of a pure computer vision segmentation method a deep-learning segmentation method that is 
    trained with a larger dataset containing coins captured under various lighting conditions and 
    camera orientation will be required. 

'''
