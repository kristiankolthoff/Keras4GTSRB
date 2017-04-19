# The German Traffic Sign Recognition Benchmark
#
# sample code for reading the traffic sign images and the
# corresponding labels
#
# example:
#            
# trainImages, trainLabels = readTrafficSigns('GTSRB/Training')
# print len(trainLabels), len(trainImages)
# plt.imshow(trainImages[42])
# plt.show()
#
# have fun, Christian

import matplotlib.pyplot as plt
import csv
import keras.utils as utils
from PIL import Image
import numpy as np
from random import shuffle

# function for reading the images
# arguments: path to the traffic sign data, for example './GTSRB/Training'
# returns: list of images, list of corresponding labels 
def readTrainData(targetWidth, targetHeight, classes, fraction=1.0, shuff=False, cat=True, 
                  rootpath='dataset/GTSRB_Img/Final_Training/Images'):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    classes = 43 if classes < 1 else 43 if classes > 43 else classes
    for c in range(0,classes):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None) #skip header of csv file
        # loop over all images in current annotations file
        for row in gtReader:
            #images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            img = Image.open(prefix + row[0])
            img = img.resize((targetWidth, targetHeight))
            images.append(np.asarray(img))
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    images = np.array(images)
    if cat:
        vals = list(zip(images, utils.to_categorical(labels, classes)))
    else:
        vals = list(zip(images, labels))
    vals = vals[:int(fraction*len(vals))]
    if shuff:
        shuffle(vals)
    zipImg, zipLabel = zip(*vals)
    return np.array(zipImg, dtype='float32'), np.array(zipLabel, dtype='float64')

def readTestData(targetWidth, targetHeight, fraction=1.0, rootpath='dataset/GTSRB_Test/Final_Test/Images'):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,2):
        prefix = rootpath + '/'
        gtFile = open(prefix + 'GT-final_test.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None) #skip header of csv file
        # loop over all images in current annotations file
        for row in gtReader:
            #images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            img = Image.open(prefix + row[0])
            img = img.resize((targetWidth, targetHeight))
            images.append(np.asarray(img))
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return (np.array(images, dtype='float32'), utils.to_categorical(labels, 43))

#testImages, testLabels = readTestData(48,48, 0.1)
#trainImages, trainLabels = readTrainData(48, 48, 43, fraction=0.2, shuff=True)
#print(len(trainLabels), len(trainImages))
#plt.imshow(trainImages[43])
#plt.imshow(trainImages[34])
#plt.show()