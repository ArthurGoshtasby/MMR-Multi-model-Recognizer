# MMR: A Multimodel Recognizer
A multimodel recognizer is developed and trained to recognize US coins.

# How to use this software
As mush as possible code and relevant resources are provided to help use of this multimodal recognizer. File Project2Qt.py contains the main module. This module implements a user interface intended to guide use of the software. 

First, create training and test datasets from the provided and possibly your own images. From each image, two datasts can be generated: a test dataset representing the detected coins and their labels, and a training dataset representing augmented coins obtained from the detected coins and their labels. Save the test and training data sets in different directories. Using the provided interface, combine the training datasets into a single larger training dataset. Also, combine the test datasets into a single larger test dataset. Make sure the training and test datasets does not overlap. 

Having the training and test datasets, train each model separately using the training dataset and save the trained models in the 'Models' directory. Modify the model's filename to reflect its recognition accuracy when tested against the test dataset. A README.txt file in the 'Models' directory explains how to do this.

Finally, use models within the 'Models' directiry to collectively reconize objects (in our case coins) in an image or a dataset.

Enjoy!

