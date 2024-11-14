# MMR-Multimodel-Recognizer
A multimodel recognizer is developed and trained to recognize US coins.

# How to use this software
I have included as mush code and other resources you need to use the multimodal recognizer. Project2Qt.py represents the main module. This module implements a user interface to guide you through the process. 

First create training and test datasets from the provided images and possibly from the images of your own. Two datasts can be created and saved from each image: a test dataset which is a dataset of detected coins anlong with their labels and a training dataset of augmented coins generated from detected coins along with their labels. Save the test and training data sets in different directories. When selecting a combination of datasets for training, make sure the dataset you use in testing do not overlap with the datasets used in training. I could not upload the combined datasets as the files are too large to upload. You should be able to combine the datasets in each directory to a larger datasets to be used for training or testing.

Having the training and test datasets, train each model separately and save the trained models in the 'Models' directory. Update the model names to reflect their recognition accuracies when testing on the same test dataset.

Finally, use the models collectively to reconize objects (in our case coins) in an image or a dataset.

Enjoy!

