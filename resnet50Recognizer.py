# Source: https://www.kaggle.com/code/suniliitb96/tutorial-keras-transfer-learning-with-resnet50
# Tutorial Keras: Transfer Learning with ResNet50
# Also see: https://www.kaggle.com/code/cokastefan/keras-resnet-50

import numpy as np
import cv2

from keras import models, layers
from keras.applications import ResNet50

#from layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization

from keras.utils import to_categorical

# Shuffle dataset and return the new dataset
def shuffleDataset(dataset):
    import random
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
            del augs[j:j+1]         # remove the jth element
            coins.append(aug)
            lbl = auglbls[j]
            del auglbls[j:j+1]
            coinlbls.append(lbl)
            i += 1

        newdataset = (coins, coinlbls)
        
    return newdataset

# Load training data
def loadData(fn,maxNoImgs):
    import pickle  
    dataset = []
    X = []
    Z = []
    with open(fn, 'rb') as fp:
        dataset = pickle.load(fp)
        dataset = shuffleDataset(dataset)
        (img,lbl) = dataset
        
        print(f"Dataset contains {len(img)} labeled coins of dimensions {img[0].shape}.")
        if maxNoImgs < 0:
            maxNoImgs = len(img)
        elif len(img) < maxNoImgs:
            maxNoImgs = len(img)
        print(f"Using {maxNoImgs} of the images.")
        for i in range(maxNoImgs):
            X.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
            Z.append(lbl[i])
        
    return X, Z


def resnet50TrainModel(datafile, coinTypes, maxNoImgs=1000, epochs=10):
    
    resnet_weights_path = 'Resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    noCointypes = len(coinTypes)

    X, Z = loadData(datafile, maxNoImgs) 

    # find image information
    img = X[0]
    noImages=len(X)
    x_train=np.array(X)
    x_train=x_train/255
    y_train=to_categorical(Z,num_classes=noCointypes)

    batch_size=noImages//64 
    if noImages < 512:          # select batch size
        batch_size = 8 

    model = models.Sequential()

    model.add(ResNet50(include_top=False, pooling='max', weights=resnet_weights_path))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(noCointypes, activation='softmax'))

    model.layers[0].trainable = False

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train,y_train, batch_size=batch_size, epochs = epochs, 
                            verbose = 2, steps_per_epoch=224 // batch_size)
    modelfn = f"{datafile+'_'+'Resnet50'+'_'+str(epochs)}.h5"
    models.save_model(model, modelfn)
    print(f"Model saved in file: {modelfn}")


