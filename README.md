# MMR: A Multimodel Recognizer
A multimodel recognizer is developed and trained to recognize US coins.

# How to use this software
As mush as possible code and relevant resources are provided to help use this multimodel recognizer. File Project2Qt.py contains the main module. This module implements a user interface that guides use of the software. 

First, create training and test datasets from the provided and possibly new similar images. From each image, two datasts can be generated: a test dataset representing the detected coins and their labels, and a training dataset representing augmented coins obtained from different orientations of the detected coins and their labels. Save the test and training data sets in different directories. A number of sample training and test datasets are provided in the Datasets directory for reference. Using the provided interface, combine the training datasets into a single larger training dataset. Also, combine the test datasets into a single larger test dataset. Make sure the training and test datasets do not overlap. 

Having the training and test datasets, train each model separately using the combined training dataset and save the trained models in the 'Models' directory. Modify each model's filename to reflect its recognition accuracy obtained when using the combined test dataset. A README.txt file in the 'Models' directory explains how to do this.

Finally, use the models within the 'Models' directiry to recognize objects (coins, in our case) in a new image or dataset.

Enjoy!

