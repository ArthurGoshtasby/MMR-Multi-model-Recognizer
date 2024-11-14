# The basics of transfer-learning are described in the following references:
# https://www.kaggle.com/code/rajmehra03/a-comprehensive-guide-to-transfer-learning
# https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import numpy as np
import LoadDisplay as ld
import TransformGeometry as tg
import CoinClass as cc
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from keras.layers import BatchNormalization
import cv2                  
import numpy as np  
import pickle
import random
from keras.applications.vgg16 import VGG16

# Shuffle dataset and return it
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

# Load training data
def loadData(fn,maxNoImgs):
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
    print(f"Using {maxNoImgs} of the images for training.")
    for i in range(maxNoImgs):
        X.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
        Z.append(lbl[i])
        
    return X, Z


def vgg16TrainModel(datafn, modelfn, weights_path, coinTypes,epochs=40, maxNoImgs=-1): # Use the entire dataset for training

    # Load data
    X, Z = loadData(datafn, maxNoImgs) 

    # find image information
    img = X[0]
    IMG_SIZE, _, _=img.shape    # dimensions of image
    noImages=len(X)             # no. images in training set
    batch_size=64        
    
    y_train=to_categorical(Z,num_classes=len(coinTypes))
    x_train=np.array(X)/255

    # Specifying the Base Model
    base_model=VGG16(include_top=False, weights=None,input_shape=X[0].shape, pooling='max')
    base_model.load_weights(weights_path)

    # Adding our Own Fully Connected Layers
    model=Sequential()
    model.add(base_model)
    model.add(Dense(256,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(len(coinTypes),activation='softmax'))
    
    early_stopping = EarlyStopping(
            min_delta=1e-4,      # minimum amount of change in monitored metric to continue
            patience=10,         # number of epochs before each monitoring event
            monitor='loss',      # use 'loss' as the metric to trigger early-stopping
            restore_best_weights=True, # use best result rather than last result
        )

    # Compile & Train the Model
    base_model.trainable=False # making the VGG model untrainable.

    # UNFREEZE THE LAST 5 LAYERS OF THE BASE MODEL
    for layer in base_model.layers[0:11]:
        layer.trainable=False
    for layer in base_model.layers[11:]:
        layer.trainable=True
   
    lr_schedule = ExponentialDecay(
        initial_learning_rate=4e-4,
        decay_steps=10,
        decay_rate=0.95
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule),loss='categorical_crossentropy',metrics=['accuracy'])
    print("\nTraining the fine-tuned VGG16 model using dataset:", datafn)
    model.fit(x_train,y_train, batch_size=batch_size,
              epochs = epochs, verbose = 2, callbacks=[early_stopping]) 
    #modelfn = f"NewModels/newallfiles_CNN_{str(epochs)}.h5"
    keras.models.save_model(model, modelfn)
    print(f"Saved model: {modelfn}")
    
    return model


# Determine values of coins in a coinsfn after reorienting each coin
# to its dominant orientation
def vgg16TestModel(datasetfn, modelfn, coinTypes):
    import pickle
    import tensorflow as tf
    
    model = tf.keras.models.Sequential()
    # if an MLP model, load it as follows
    # model = pickle.load(open(modelfn, "rb"))
    # otherwise, load the model as follows
    model = tf.keras.models.load_model(modelfn, compile = False)
    
    vgg16TestThisModel(datasetfn, model, coinTypes)

def vgg16TestThisModel(datasetfn, model1, coinTypes):
    import pickle
    import tensorflow as tf
    
    model = tf.keras.models.Sequential()
    model = model1
    
    co = cc.Coins()
    dataset = co.loadDataset(datasetfn)
    (coins, lbls) = dataset

    rcoins = []
    for i in range(len(coins)):
        coin = coins[i]
        lor = tg.findDominantOrientations(coin, 1) # find coin's dominant orientation
        rcoin = tg.rotateImage(coin,lor[0])
        rcoins.append(rcoin)
        
    val_images = []
    for i in range(len(rcoins)):
        val_images.append(cv2.cvtColor(rcoins[i], cv2.COLOR_BGR2RGB))
        
    val_images = np.array(val_images)/255
    
    predictions = model.predict(val_images)

    n = len(rcoins) 
    m = 0
    for i in range(n):
        ctype = np.argmax(predictions[i])
        title = f"{i}:{coinTypes[ctype]}"
        if ctype == lbls[i]:
            m += 1
            print(f"Correct: {coinTypes[ctype]} -> {coinTypes[lbls[i]]}")
        else: 
            print(f"False:   {coinTypes[ctype]} -> {coinTypes[lbls[i]]}")
    
    acc = m/n
    print(f"Accuracy: {acc:0.2f}")
    return



# Determine the total value of coins in an image
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

# Determine value of coins in an image, after reorienting each coin to
# its dominant orientation. It is assumed that the model has been trained
# using a dataset where each coin is reoriented to a fixed number (such as 5) of its
# most dominant orientations.
def findTotalCoinValueNew(imgfn, modelfn, coinTypes, coinValues):
    import CoinClass as cc
    
    model = keras.models.Sequential()
    model = keras.models.load_model(modelfn, compile = False)
    
    cp = cc.Coins()
    coins = cp.extractCoins(imgfn)
    
    rcoins = []
    for i in range(len(coins)):
        coin = coins[i]
        lor = tg.findDominantOrientations(coin, 1)  # find coin's dominant orientation
        rcoin = tg.rotateImage(coin,lor[0])         # reorient coint to its dominant orientation
        rcoins.append(rcoin)
    
    val_images = []
    for i in range(len(rcoins)):
        val_images.append(cv2.cvtColor(rcoins[i], cv2.COLOR_BGR2RGB))
        
    val_images = np.array(val_images)
    val_images = val_images/255
    
    predictions = model.predict(val_images)

    location = [2, 2]
    n = len(rcoins) 
    totalvalue = 0.0
    print("Press a key on keyboard to move to next coin.")
    for i in range(n):
        ctype = np.argmax(predictions[i])
        title = f"{i}:{coinTypes[ctype]}"
        print(f"Coin type: {coinTypes[ctype]}, coin value: ${coinValues[ctype]/100.0:.2f}")
        ld.displayScaledImage(title,coins[i],2, location)
        totalvalue += coinValues[ctype]/100.0
        
    print(f"Total value of coins: ${totalvalue:.2f}")
    
    cv2.destroyAllWindows()
    
    return totalvalue

'''
# The basics of transfer-learning are described in the following references:
# https://www.kaggle.com/code/rajmehra03/a-comprehensive-guide-to-transfer-learning
# https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
import numpy as np
import LoadDisplay as ld
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from keras.layers import BatchNormalization
import cv2                  
import numpy as np  
import pickle
import random
from keras.applications.vgg16 import VGG16

# Shuffle dataset and return it
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

# Load training data
def loadData(fn,maxNoImgs):
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
    print(f"Using {maxNoImgs} of the images for training.")
    for i in range(maxNoImgs):
        X.append(cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB))
        Z.append(lbl[i])
        
    return X, Z


def vgg16TrainModel(datafn, weights_path, coinTypes,epochs=40, maxNoImgs=-1): # Use the entire dataset for training
    print("Creating a VGG16 Recognizer model.")
    # Load data
    coins, coinlbls = loadData(datafn, maxNoImgs) 

    # find image information
    IMG_SIZE, _, _=coins[0].shape    # dimensions of image
    
        
    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    # Split data into training and validation
    for i in range(len(coins)):
        # convert opencv BGR images to RGB images
        if i%11 == 0:
            valid_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
            valid_labels.append(coinlbls[i])
        else:
            train_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
            train_labels.append(coinlbls[i])
            
    # Convert lists to arrays as the recognizer requires them
    train_images = np.array(train_images)
    train_images = train_images/255 - 0.5
    train_labels = np.array(train_labels)
    train_labels = to_categorical(train_labels,num_classes=len(coinTypes))
    valid_images = np.array(valid_images)
    valid_images = valid_images/255 - 0.5
    valid_labels = np.array(valid_labels)
    valid_labels = to_categorical(valid_labels,num_classes=len(coinTypes))

    noImages=len(train_images)             # no. images in training set
    batch_size=noImages//64
    if noImages < 512:          # batch size
        bach_size = 16  
        
    # Specifying the Base Model
    base_model=VGG16(include_top=False, weights=None,input_shape=coins[0].shape, pooling='max')
    base_model.load_weights(weights_path)

    # Adding our Own Fully Connected Layers
    model=Sequential()
    model.add(base_model)
    model.add(Dense(1024,activation='elu'))
    #model.add(BatchNormalization())
    model.add(Dense(len(coinTypes),activation='softmax'))
    
    early_stopping = EarlyStopping(
            min_delta=1e-4,      # minimum amount of change in monitored metric to continue
            patience=10,         # number of epochs before each monitoring event
            monitor='loss',      # use 'loss' as the metric to trigger early-stopping
            restore_best_weights=True, # use best result rather than last result
        )

    # Compile & Train the Model
    base_model.trainable=False # making the VGG model untrainable.

    # UNFREEZE THE LAST 5 LAYERS OF THE BASE MODEL
    for layer in base_model.layers[0:11]:
        layer.trainable=False
    for layer in base_model.layers[11:]:
        layer.trainable=True
   
    lr_schedule = ExponentialDecay(
        initial_learning_rate=4e-4,
        decay_steps=10,
        decay_rate=0.95
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule),loss='categorical_crossentropy',metrics=['accuracy'])
    print("\nTraining the modified VGG16 model using dataset:", datafn)
    model.fit(train_images, train_labels,validation_data = (valid_images, valid_labels),
              batch_size=batch_size,
              epochs = epochs, verbose = 2, callbacks=[early_stopping]) 
    modelfn = f'NewModels/newallfiles_vgg16_{str(epochs)}.h5'
    keras.models.save_model(model, modelfn)
    print(f"Saved model: {modelfn}")
    
    return modelfn


# Determine the total value of coins in an image
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
    val_images = val_images/255 - 0.5
    
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



'''