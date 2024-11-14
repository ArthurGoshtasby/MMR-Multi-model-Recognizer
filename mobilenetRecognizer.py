# The recognizer in this module is a retrained version of MobileNet
# described here: https://arxiv.org/abs/1704.04861   
# It was designed to recognize birds as described below:
# https://www.kaggle.com/code/umairshahpirzada/birds-20-classification-mobilenet


import cv2                  
import pickle
import random
import numpy as np 
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.layers import Input
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from keras.applications.mobilenet import MobileNet
import LoadDisplay as ld

import os
import tensorflow as tf  
tf.get_logger().setLevel('ERROR')   # suppress tensorflow warnings

def mobilenet(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    base_model = MobileNet(include_top=False, input_tensor=input_tensor)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)

    return model

# Reshuffle the dataset and return it
def shuffleDataset(dataset):
    (augs, auglbls) = dataset
    newdataset = []
    n = len(augs)
    if n == 0:
        print("dataset is empty!")
    else:
        coins = []
        coinlbls = []
        i = 0
        while i < n:
            m = len(augs)-1
            j = random.randint(0,m)
            aug = augs[j]           # Get the jth element
            del augs[j:j+1]         # Remove the jth element
            coins.append(aug)
            lbl = auglbls[j]
            del auglbls[j:j+1]
            coinlbls.append(lbl)
            i += 1

        newdataset = (coins, coinlbls)
        
    return newdataset

# Load the training dataset
def loadData(fn,maxNoImgs):
    dataset = []
    X = []
    Z = []
    with open(fn, 'rb') as fp:
        dataset = pickle.load(fp)
    dataset = shuffleDataset(dataset)
    (img,lbl) = dataset
        
    if maxNoImgs < 0:
        maxNoImgs = len(img)
    elif len(img) < maxNoImgs:
        maxNoImgs = len(img)
    print(f"Using {maxNoImgs} of the images for training.")
    for i in range(maxNoImgs):
        X.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
        Z.append(lbl[i])
    print(f"Dataset contains {len(img)} labeled coins of dimensions {img[0].shape}.")
        
    return X, Z


def mobilenetTrainModel(datafn, modelfn, coinTypes, epochs=40, maxNoImgs=-1):
    print("Creating a Mobilenet Recognizer.")
    # Load data
    X, Z = loadData(datafn, maxNoImgs) 

    noImages=len(X)             # no. images in training set
    
    batch_size=noImages//64     # select batch size
    if noImages < 512:          
        bach_size = 8         
    
    y_train=to_categorical(Z,num_classes=len(coinTypes))
    x_train=np.array(X)/255 - 0.5
    input_shape = x_train[0].shape

    num_classes = len(coinTypes)
    
    early_stopping = EarlyStopping(
            min_delta=1e-4,     # minimum amount of change in monitored metric to continue
            patience=10,        # number of epochs before each monitoring event
            monitor='loss',     # use 'loss' as the metric to trigger early-stopping
            restore_best_weights=True) # use best result rather than last result

    model = mobilenet(input_shape, num_classes)
    
    lr_schedule = ExponentialDecay(
        initial_learning_rate=4e-4,
        decay_steps=10,
        decay_rate=0.95
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs = epochs, verbose = 2, callbacks=[early_stopping])
    
    #modelfn = f'NewModels/newallfiles_mobilenet_{str(epochs)}.h5'
    keras.models.save_model(model, modelfn)
    print(f"Saved model in file: {modelfn}")
    
    return model


# Determine value of coins in an image
def findTotalCoinValue(imgfn, modelfn, coinTypes, coinValues):
    import CoinClass as cc
    
    model = keras.models.Sequential()
    model = keras.models.load_model(modelfn, compile = False)
    
    cp = cc.Coins()
    coins = cp.extractCoins(imgfn)
    
    val_images = []
    for i in range(len(coins)):
        val_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
        
    val_images = np.array(val_images)
    val_images = val_images/255
    
    predictions = model.predict(val_images)

    location = [2, 2]
    n = len(coins) 
    totalvalue = 0.0
    for i in range(n):
        ctype = np.argmax(predictions[i])
        title = f"{i}:{coinTypes[ctype]}"
        print(f"Coin type: {coinTypes[ctype]}, coin value: ${coinValues[ctype]/100.0:.2f}")
        ld.displayScaledImage(title,coins[i],2, location)
        totalvalue += coinValues[ctype]/100.0
        
    print(f"Total value of coins: ${totalvalue:.2f}")
    
    cv2.destroyAllWindows()
    
    return totalvalue


