First combine the datasets in the Datasets/Training directory to a file say 'trainDataset'. Then use 'trainDataset' as the training dataset to train each of the 4 models. Save the created models in this directory.

Combines datasets not used above to create a 'testDataset' to be used to find the accuracy of each model.

Use the 'testDataset' to find the accuracy of each model. Precede the name of a created model with 100xaccuracy (percent correct recognition). For example, if accuracy of a model is 0.81 and model name is 'modelname.h5', change model name to '81_modelname.h5'. Recognition accuracy of each model is used in decision making when two or more models are used to recognize objects (coins in our case).

When extracting coins in an image, the extracted coins represent a dataset that should be used to test a model's accuracy, while augmented coins created from the detected coins should be used to train a model. A training dataset contains 5 different orientations of a detected coin, while a test dataset contains only the detected coins in their original orientations.