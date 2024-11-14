
# This module contains a few simple coin recognizers and related methods


import LoadDisplay as ld
import CoinClass as cc
import TransformGeometry as tg
import numpy as np
import keras
from keras.utils import to_categorical
from keras import layers, models
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.optimizers.schedules import ExponentialDecay
from sklearn.neural_network import MLPClassifier
import pickle
import random
import cv2
import os

coinTypes42 = ["DoFJMa", "DoFJMo", "DiF", "PF", "NFN", "NFO", "QFO", "QFN",
                "DoB", "DiB", "PBM", "PBH", "PBS", "PBO", "NB", "QBNH", "QBMI",
                "QBOE", "QBAZ", "QBGAB", "QBND", "QBKY", "QBTX", "QBDE", "QBLA",
                "QBNYL", "QBNYS", "QBFL", "QBMD", "QBIA", "QBVT", "QBND", "QBNW",
                "QBNJ", "QBWV", "QBWI", "QBUT", "QBNV", "QBSC", "QBOK", "QBGAP",
                "QBVA"] 
coinValues42 = [100, 100, 10, 1, 5, 5, 25, 25,
                 100, 10, 1, 1, 1, 1, 5, 25, 25,
                 25, 25, 25, 25, 25, 25, 25, 25,
                 25, 25, 25, 25, 25, 25, 25, 25,
                 25, 25, 25, 25, 25, 25, 25, 25,
                 25] 
coinTypes5 = ["Penny", "Nickle", "Dime", "Quarter", "Dollar"]
coinValues5 = [1, 5, 10, 25, 100]

# Shuffle dataset to mix coins
def shuffleDataset(dataset):
    (augs, auglbls) = dataset
    newdataset = []
    coins = []
    coinlbls = []
    n = len(augs)
    if n == 0:
        print("dataset is empty!")
    else:
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

# Create training data from dataset
def getTrainingData(dataset):
    (coins,coinlbls) = dataset    
    ntrain = len(coins)        # Number of training images

    print(f"Dataset contains {ntrain} labeled color images. ")
    print(f"Image shape: {coins[0].shape}")
        
    train_images = []
    train_labels = []
    for i in range(ntrain):
        # convert opencv BGR images to RGB images
        train_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
        train_labels.append(coinlbls[i])
            
    # Convert lists to arrays as the recognizers require them
    train_images = np.array(train_images)
    train_images = train_images/255
    train_labels = np.array(train_labels)
        
    return (train_images, train_labels)

# Train a CNN recognizer
def CNNRecognizer(datafn, modelfn, coinTypes, epochs = 40): 
    print("Creating a CNN Recognizer.")
    dataset = []
    with open(datafn, 'rb') as fp:
        dataset = pickle.load(fp)
        
    # Shuffle dataset, in case it is not
    dataset = shuffleDataset(dataset)
        
    (coins,coinlbls) = dataset    
    ntrain = len(coins)        # Number of training images

    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    # Split data into training and validation
    for i in range(ntrain):
        # convert opencv BGR images to RGB images
        if i%19 == 0:
            valid_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
            valid_labels.append(coinlbls[i])
        else:
            train_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
            train_labels.append(coinlbls[i])
            
    # Convert lists to arrays as the recognizer requires them
    train_images = np.array(train_images)/255 - 0.5
    train_labels = np.array(train_labels)
    train_labels = to_categorical(train_labels,num_classes=len(coinTypes))
    valid_images = np.array(valid_images)/255 - 0.5
    valid_labels = np.array(valid_labels)
    valid_labels = to_categorical(valid_labels,num_classes=len(coinTypes))
    
    print(f"Dataset contains {ntrain} labeled color images, of which ")
    print(f"{len(train_images)} are used for testing and {len(valid_images)} are used for validation.")
    print(f"Image shape: {coins[0].shape}")
        
    # Add early stopping capability to the learning process.
    early_stopping = EarlyStopping(
        min_delta=1e-4,      # minimum amount of change in monitored metric to continue
        patience=10,         # number of epochs before monitoring
        monitor='loss',      # use 'loss' as the metric to trigger early-stopping
        restore_best_weights=True) # use best result rather than last result

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='elu', input_shape=train_images[0].shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='elu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='elu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='elu'))
    model.add(layers.Dense(len(coinTypes), activation='softmax')) # Make outputs represent probabilities

    lr_schedule = ExponentialDecay(
        initial_learning_rate=4e-4,
        decay_steps=10,
        decay_rate=0.95
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, 
              validation_data=(valid_images, valid_labels), 
              batch_size=64, epochs=epochs, 
              callbacks=[early_stopping], verbose=2)
    models.save_model(model, modelfn)
    print(f"Model saved in file: {modelfn}")
    
    return model

# Train a basic (densely connected) recognizer
def BasicRecognizer(datafn, modelfn, coinTypes, epochs = 20): 
    print("Creating a Basic (fully-connected) Recognizer.")
    dataset = []
    with open(datafn, 'rb') as fp:
        dataset = pickle.load(fp)
        
    # Shuffle dataset, in case it is not
    dataset = shuffleDataset(dataset)
    (coins, coinlbls) = dataset
        
    (coins,coinlbls) = dataset    
    ntrain = len(coins)        # Number of training images

    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    # Split data into training and validation
    for i in range(ntrain):
        # convert opencv BGR images to RGB images
        if i%19 == 0:
            valid_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
            valid_labels.append(coinlbls[i])
        else:
            train_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
            train_labels.append(coinlbls[i])
            
    # Convert lists to arrays as the recognizer requires them
    train_images = np.array(train_images)/255 -0.5
    train_labels = np.array(train_labels)
    train_labels = to_categorical(train_labels,num_classes=len(coinTypes))
    valid_images = np.array(valid_images)/255 - 0.5
    valid_labels = np.array(valid_labels)
    valid_labels = to_categorical(valid_labels,num_classes=len(coinTypes))

    print(f"Dataset contains {ntrain} labeled color images, of which ")
    print(f"{len(train_images)} are used for testing and {len(valid_images)} are used for validation.")
    print(f"Image shape: {coins[0].shape}")
            
    # Add early stopping capability to the learning process.
    early_stopping = EarlyStopping(
        min_delta=1e-4,     # minimum amount of change in monitored metric to continue
        patience=10,         # number of epochs before monitoring
        monitor='loss',     # use 'loss' as the metric to trigger early-stopping
        restore_best_weights=True) # use best result rather than last result

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=train_images[0].shape),        
        keras.layers.Dense(512, activation='elu'),
        keras.layers.Dense(512, activation='elu'),
        keras.layers.Dense(512, activation='elu'),
        keras.layers.Dense(512, activation='elu'),
        keras.layers.Dense(512, activation='elu'),
        keras.layers.Dense(len(coinTypes), activation='softmax')]) # Make outputs represent probabilities

    lr_schedule = ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_steps=10,
        decay_rate=0.95
    )
    model.compile(optimizer=Adam(learning_rate=lr_schedule), 
                  loss='categorical_crossentropy',  metrics=['accuracy'])
    
    # Add early stopping capability to the learning process.
    early_stopping = EarlyStopping(
        min_delta=1e-5,     # minimum amount of change in monitored metric to continue
        patience=10,        # number of epochs before each monitoring event
        monitor='loss',     # use 'loss' as the metric to trigger early-stopping
        restore_best_weights=True, # use best result rather than last result
    )

    model.fit(train_images, train_labels, 
              validation_data=(valid_images, valid_labels),
              batch_size=64, epochs=epochs, callbacks=[early_stopping], verbose = 2)
        
    models.save_model(model, modelfn)
    print(f"Saved Model in file: {modelfn}")
        
    return model

# Train an MLP recognizer
def MLPRecognizer(datafn, coinTypes, epochs = 20): 
    dataset = []
    with open(datafn, 'rb') as fp:
        dataset = pickle.load(fp)
        
    # Shuffle dataset, in case it is not
    dataset = shuffleDataset(dataset)
    (coins, coinlbls) = dataset
        
    (coins,coinlbls) = dataset    
    ntrain = len(coins)        # Number of training images

    print(f"Dataset contains {ntrain} labeled color images. ")
    print(f"Image shape: {coins[0].shape}")
    train_images = []
    train_labels = []
    for i in range(ntrain):
        # convert opencv BGR images to RGB images
        train_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
        train_labels.append(coinlbls[i])
            
    # Convert lists to arrays as the recognizer requires them
    train_images = np.array(train_images)
    train_images = train_images/255.0
    (rows,cols,bands) = train_images[0].shape
    imgsize = rows*cols*bands
    train_images = np.reshape(train_images, (len(train_images), imgsize))
    train_labels = np.array(train_labels)
    train_labels = to_categorical(train_labels,num_classes=len(coinTypes))
        
    # Add early stopping capability to the learning process.
    MLP = MLPClassifier(
        activation = 'relu',                #default
        hidden_layer_sizes=(64,64,64,64,),  #default = (100,),
        early_stopping = True,              #default = False
        max_iter=epochs,                    #default = 200
        tol = 0.001,                        #default = 1e-4
        n_iter_no_change = 5,               #default = 10
        # When early_stopping is set to True, use 10% of training data to 
        # find validation score and use that to stop early when the validation
        # score does not change by more than tol within n_iter_no_change iterations.
        solver="adam",                      #default
        verbose=True)                       #defualt = False

    model = MLP.fit(train_images, train_labels)     
    modelfn = f"{datafn}_MLP_{str(epochs)}.mlp"
    pickle.dump(model, open(modelfn, "wb"))
    print(f"Model saved in file: {modelfn}")
        
    return modelfn

# Let multiple models vote for a coin type, then choose the 
# label voted by most models. Provide either an image filename or
# a filename containing a list of coins. If both are provided, the
# filename cointaining a list of coins will be used.
def recognizeUsingMultipleModels(modelsdir, coinTypes, coinValues, imagefn = "", coinsfn=""):
    coins = []
    if imagefn == "" and coinsfn == "":
        print("You must provide either an image filename or a coins filename.")
        return
    elif imagefn == "":
        with open(coinsfn, 'rb') as fp:
            coins = pickle.load(fp)
    else:
       import CoinClass as cc
       cp = cc.Coins()
       coins = cp.extractCoins(imagefn)
      
    ncoins = len(coins)
    val_images = []
    for i in range(ncoins):
        val_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
        
    val_images = np.array(val_images)
    val_images = val_images/255 - 0.5
    
    modelfns = os.listdir(modelsdir)

    nmodels = len(modelfns)
    models = []
    for j in range(nmodels):
        modelfn = modelsdir+'/'+modelfns[j]
        model = keras.models.Sequential()
        # if an MLP model, load it as follows
        # model = pickle.load(open(modelfn, "rb"))
        # otherwise, load the model as follows
        model = keras.models.load_model(modelfn, compile = False)
        models.append(model)
    
    print(f"Recognizing coins using {len(models)} models.")
    location = [2, 2]
    # Method1:
    # for each coin
    totalvalue = 0
    notsure = False
    
    # for each coin
    for i in range(ncoins):
        votes = np.zeros(5)
        # for each model
        for j in range(nmodels):
            model = models[j]
            # find predictions
            predictions = model.predict(val_images, verbose=0)
            lbl = np.argmax(predictions[i])
            votes[lbl] += 1
            
        print(votes)
        lbl1 = np.argmax(votes)
        val1 = votes[lbl1]
        votes[lbl1] = 0
        lbl2 = np.argmax(votes)
        val2 = votes[lbl2]
        print(val1,val2)
        if val1 == val2:
            notsure = True
            title1 = coinTypes[lbl1]
            title2 = coinTypes[lbl2]
            print(f"Coin {i} has either label {lbl1}: ({title1}) or label {lbl2}: ({title2})")
            ld.displayScaledImage(f'{title1} or {title2}',coins[i],2, location)
        else:
            title1 = coinTypes[lbl1]
            print(f"Coin {i} has label {lbl1}: {title1}")
            ld.displayScaledImage(title1,coins[i],2, location)

        totalvalue += coinValues[lbl1]
    cv2.destroyAllWindows()

    totalvalue /= 100.0
    if notsure:
        print(f"Not sure, total value may be ${totalvalue:.2f}")
        return -totalvalue
    else:
        print(f"Total value of coins: ${totalvalue:.2f}")
        return totalvalue
    
# Let multiple models vote for a coin label, then choose the 
# label voted by most models. Provide either an image filename or
# a filename containing a dataset. If both are provided, the
# filename cointaining an image of coins will be used. It is assumed
# that the models are trained on coins reoriented to 
# dominant orientations of coins
def recognizeUsingMultipleModelsNew(modelsdir, coinTypes, coinValues, imagefn = "", datasetfn=""):
    import CoinClass as cc
    coins = []
    labels = []
    if imagefn == "" and datasetfn == "":
        print("You must provide either an image filename or a dataset filename.")
        return
    elif imagefn == "":
        cp = cc.Coins()
        dataset = cp.loadDataset(datasetfn)
        (coins,labels) = dataset
    else:
       cp = cc.Coins()
       coins = cp.extractCoins(imagefn)
       
    rcoins = []
    for i in range(len(coins)):
        coin = coins[i]
        lor = tg.findDominantOrientations(coin, 1) # find the dominant orientation
        rcoin = tg.rotateImage(coin,lor[0])
        rcoins.append(rcoin)
      
    ncoins = len(rcoins)
    val_images = []
    for i in range(ncoins):
        val_images.append(cv2.cvtColor(rcoins[i], cv2.COLOR_BGR2RGB))
    
    modelfns = os.listdir(modelsdir)
    nmodels = len(modelfns)
    models = []
    weights = []
    vgg16 = []
    for j in range(nmodels):
        modelfn = modelfns[j]
        words = modelfn.split('_')
        weight = int(words[0])/100
        weights.append(weight)
        word = words[1]
        if word[4]=='G' and word[5]=='G' and word[6]=='V':
            vgg16.append(True)
        else:
            vgg16.append(False)
        model = keras.models.Sequential()
        # if an MLP model, load it as follows
        # model = pickle.load(open(modelfn, "rb"))
        # otherwise, load the model as follows
        modelfn = modelsdir+'/'+modelfns[j]
        model = keras.models.load_model(modelfn, compile = False)
        models.append(model)

    print("Using these weights for the models:", weights)
    print(f"Recognizing coins using {len(models)} models.")
    location = [2, 2]
    # Method1:
    # for each coin
    totalvalue = 0
    notsure = False
    
    # for each coin
    valimages1 = val_images = np.array(val_images)/255 - 0.5
    valimages2= val_images = np.array(val_images)/255
    valimages = valimages1
    ncorrect = 0
    for i in range(ncoins):
        votes = np.zeros(5)
        # for each model
        for j in range(nmodels):
            if vgg16[j]:
                valimages = valimages2
            else:
                valimages = valimages1
            model = models[j]
            # find predictions
            predictions = model.predict(valimages, verbose=0)
            lbl = np.argmax(predictions[i])
            votes[lbl] += weights[j]
            
        print(f"Votes: [{votes[0]:.2f}, {votes[1]:.2f}, {votes[2]:.2f}, {votes[3]:.2f}, {votes[4]:.2f}]")
        lbl1 = np.argmax(votes)
        val1 = votes[lbl1]
        votes[lbl1] = 0
        lbl2 = np.argmax(votes)
        val2 = votes[lbl2]
        #print(f"{val1:.2f}, {val2:.2f}")
        if val1 == val2:
            notsure = True
            title1 = coinTypes[lbl1]
            title2 = coinTypes[lbl2]
            print(f"Coin {i} has either label {lbl1}: ({title1}) or label {lbl2}: ({title2})")
            if labels == []:
                ld.displayScaledImage(f'{title1} or {title2}',coins[i],2, location)
        else:
            title1 = coinTypes[lbl1]
            print(f"Coin {i} has label {lbl1}: {title1}")
            if labels == []:
                ld.displayScaledImage(title1,coins[i],2, location)
            else:
                print("True label:",labels[i])
                if labels[i] == lbl1:
                    ncorrect += 1

        if labels == []:
            totalvalue += coinValues[lbl1]

    if labels == []:
        cv2.destroyAllWindows()
        totalvalue /= 100
        if notsure:
            print(f"Not sure, total value may be ${totalvalue:.2f}")
            return -totalvalue
        else:
            print(f"Total value of coins in image: ${totalvalue:.2f}")
            return totalvalue
    else:
        print(f"Accuracy: {ncorrect/ncoins:.2f}")

# Determine value of coins in a coinsfn
def validatePredictions(datasetfn, modelfn, coinTypes):
    import pickle
    import tensorflow as tf
    
    model = keras.models.Sequential()
    # if an MLP model, load it as follows
    # model = pickle.load(open(modelfn, "rb"))
    # otherwise, load the model as follows
    model = keras.models.load_model(modelfn, compile = False)
    
    co = cc.Coins()
    dataset = co.loadDataset(datasetfn)
    (coins, lbls) = dataset
    val_images = []
    for i in range(len(coins)):
        val_images.append(cv2.cvtColor(coins[i], cv2.COLOR_BGR2RGB))
        
    val_images = np.array(val_images)
    val_images = val_images/255 - 0.5
    # If using an MLP model, execute the next 3 statements
    # (rows,cols,bands) = val_images[0].shape
    # imgsize = rows*cols*bands
    # val_images = np.reshape(val_images, (len(val_images), imgsize))
    
    predictions = model.predict(val_images)

    location = [2, 2]
    n = len(coins) 
    for i in range(n):
        ctype = np.argmax(predictions[i])
        title = f"{i}:{coinTypes[ctype]}"
        if ctype == lbls[i]:
            print(f"Correct: {coinTypes[ctype]} -> {coinTypes[lbls[i]]}")
        else: 
            print(f"False:   {coinTypes[ctype]} -> {coinTypes[lbls[i]]}")
    
    return

# Determine values of coins in a coinsfn after reorienting each coin
# to its dominant orientation
def testModel(datasetfn, modelfn, coinTypes):
    import pickle
    import tensorflow as tf
    
    model = tf.keras.models.Sequential()
    # if an MLP model, load it as follows
    # model = pickle.load(open(modelfn, "rb"))
    # otherwise, load the model as follows
    model = tf.keras.models.load_model(modelfn, compile = False)
    testThisModel(datasetfn, model, coinTypes)

def testThisModel(datasetfn, model1, coinTypes):
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
        
    val_images = np.array(val_images)/255 - 0.5
    
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

# Determine value of coins in an image assuming the model has been 
# trained using an augmented dataset obtrained by rotating each coin
# about its center by a fixed increment, such as 5 degrees.
def findTotalCoinValue(imgfn, modelfn, coinTypes, coinValues):
    import CoinClass as cc
    
    model = keras.models.Sequential()
    model = keras.models.load_model(modelfn, compile = False)
    
    cp = cc.Coins()
    rcoins = cp.extractCoins(imgfn)
    
    val_images = []
    for i in range(len(rcoins)):
        val_images.append(cv2.cvtColor(rcoins[i], cv2.COLOR_BGR2RGB))
        
    val_images = np.array(val_images)
    val_images = val_images/255 - 0.5
    
    predictions = model.predict(val_images)

    location = [2, 2]
    n = len(rcoins) 
    totalvalue = 0.0
    print("Press a key on keyboard to move to next coin.")
    for i in range(n):
        ctype = np.argmax(predictions[i])
        title = f"{i}:{coinTypes[ctype]}"
        print(f"Coin type: {coinTypes[ctype]}, coin value: ${coinValues[ctype]/100.0:.2f}")
        ld.displayScaledImage(title,rcoins[i],2, location)
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
        
    val_images = np.array(val_images)/255 - 0.5
    
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