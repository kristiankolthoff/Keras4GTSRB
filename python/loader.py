#Partly taken from The German Traffic Sign Recognition Benchmark template
import csv
import cv2
from PIL import ImageOps as ops
import keras.utils as utils
from PIL import Image
import numpy as np
from random import shuffle

def readTrainData(targetWidth, targetHeight, classes, fraction=1.0, shuff=False, cat=True, grayscale = False,
                  normalization = "none",
                  rootpath='../dataset/GTSRB_Img/Final_Training/Images'):
    images = [] # images
    labels = [] # corresponding labels
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
            if(grayscale):
                img = img.convert('L', (0.2989, 0.5870, 0.1140, 0))
            img = img.crop((int(row[3]), int(row[4]), int(row[5]), int(row[6])))
            img = img.resize((targetWidth, targetHeight))
             #Normalize final image with specific method for better contrast
            if(normalization == "autocontrast"):
                img = ops.autocontrast(img)
            elif(normalization == "histequi"):
                img = ops.equalize(img)
            elif(normalization == "histequi_cv2_yuv"):
                imgyuv = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2YUV)
                imgyuv[:,:,0] = cv2.equalizeHist(imgyuv[:,:,0])
                img = cv2.cvtColor(imgyuv, cv2.COLOR_YUV2BGR)
                img = Image.fromarray(img)
            elif(normalization == "histequi_cv2_rgb"):
                imgrgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                for c in range(0, 2):
                    imgrgb[:,:,c] = cv2.equalizeHist(imgrgb[:,:,c])
                img = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            elif(normalization == "adahistequi_cv2_yuv"):
                clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
                imgyuv = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2YUV)
                imgyuv[:,:,0] = clahe.apply(imgyuv[:,:,0])
                img = cv2.cvtColor(imgyuv, cv2.COLOR_YUV2BGR)
                img = Image.fromarray(img)
            elif(normalization == "adahistequi_cv2_rgb"):
                clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
                imgrgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                for c in range(0, 2):
                    imgrgb[:,:,c] = clahe.apply(imgrgb[:,:,c])
                img = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            if(grayscale):
                images.append(np.asarray(img).reshape(targetWidth, targetHeight, 1))
            else:
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

def readTestData(targetWidth, targetHeight, fraction=1.0, grayscale = False, normalization = "None",
                 rootpath='../dataset/GTSRB_Test/Final_Test/Images'):
    images = [] # images
    labels = [] # corresponding labels
    for c in range(0,1):
        prefix = rootpath + '/'
        gtFile = open(prefix + 'GT-final_test.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None) #skip header of csv file
        # loop over all images in current annotations file
        for row in gtReader:
            #images.append(plt.imread(prefix + row[0])) # the 1th column is the filename
            img = Image.open(prefix + row[0])
            if(grayscale):
                img = img.convert('L', (0.2989, 0.5870, 0.1140, 0))
            img = img.crop((int(row[3]), int(row[4]), int(row[5]), int(row[6])))
            img = img.resize((targetWidth, targetHeight))
             #Normalize final image with specific method for better contrast
            if(normalization == "autocontrast"):
                img = ops.autocontrast(img)
            elif(normalization == "histequi"):
                img = ops.equalize(img)
            elif(normalization == "histequi_cv2_yuv"):
                imgyuv = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2YUV)
                imgyuv[:,:,0] = cv2.equalizeHist(imgyuv[:,:,0])
                img = cv2.cvtColor(imgyuv, cv2.COLOR_YUV2BGR)
                img = Image.fromarray(img)
            elif(normalization == "histequi_cv2_rgb"):
                imgrgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                for c in range(0, 2):
                    imgrgb[:,:,c] = cv2.equalizeHist(imgrgb[:,:,c])
                img = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            elif(normalization == "adahistequi_cv2_yuv"):
                clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
                imgyuv = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2YUV)
                imgyuv[:,:,0] = clahe.apply(imgyuv[:,:,0])
                img = cv2.cvtColor(imgyuv, cv2.COLOR_YUV2BGR)
                img = Image.fromarray(img)
            elif(normalization == "adahistequi_cv2_rgb"):
                clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
                imgrgb = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                for c in range(0, 2):
                    imgrgb[:,:,c] = clahe.apply(imgrgb[:,:,c])
                img = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
            if(grayscale):
                images.append(np.asarray(img).reshape(targetWidth, targetHeight, 1))
            else:
                images.append(np.asarray(img))
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return (np.array(images, dtype='float32'), utils.to_categorical(labels, 43))


def readHaarFeaturesTrainData(rootpath='../dataset/GTSRB_Haar/Final_Training/Haar', normalize=False):
    features = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = '../dataset/GTSRB_Img/Final_Training/Images/' + format(c, '05d') + '/' # subdirectory for class
        prefix2 = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None) #skip header of csv file
        # loop over all images in current annotations file
        for row in gtReader:
            haarFile = open(prefix2 + row[0].split('.')[0] + '.txt')
            haarReader = csv.reader(haarFile, delimiter=';')
            haarFeature = []
            for feat in haarReader:
                haarFeature.append(float(feat[0]))
            features.append(np.asarray(haarFeature))
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return (np.array(features, dtype='float32'), utils.to_categorical(labels, 43))
        
def readHOGFeaturesTrainData(rootpath='../dataset/GTSRB_Final_Training_HOG/GTSRB/Final_Training/HOG', 
                             set='HOG_02', normalize=False):
    features = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    for c in range(0,43):
        prefix = '../dataset/GTSRB_Img/Final_Training/Images/' + format(c, '05d') + '/' # subdirectory for class
        prefix2 = rootpath + '/' + set + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader, None) #skip header of csv file
        # loop over all images in current annotations file
        for row in gtReader:
            hogFile = open(prefix2 + row[0].split('.')[0] + '.txt')
            hogReader = csv.reader(hogFile, delimiter=';')
            hogFeature = []
            for feat in hogReader:
                hogFeature.append(float(feat[0]))
            hogFile.close()
            features.append(np.asarray(hogFeature))
            labels.append(row[7]) # the 8th column is the label
        gtFile.close()
    return (np.array(features, dtype='float32'), utils.to_categorical(labels, 43)) 

def readHOGFeaturesTestData(rootpath='../dataset/GTSRB/Final_Test/HOG', 
                             set='HOG_02', normalize=False):
    features = [] # images
    labels = [] # corresponding labels
    # loop over all 42 classes
    prefix = '../dataset/GTSRB_Test/Final_Test/Images/'
    gtFile = open(prefix + 'GT-final_test.csv') # annotations file
    prefix2 = rootpath + '/' + set + '/' # subdirectory for class
    gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
    next(gtReader, None) #skip header of csv file
    # loop over all images in current annotations file
    for row in gtReader:
        hogFile = open(prefix2 + row[0].split('.')[0] + '.txt')
        hogReader = csv.reader(hogFile, delimiter=';')
        hogFeature = []
        for feat in hogReader:
            hogFeature.append(float(feat[0]))
        hogFile.close()
        features.append(np.asarray(hogFeature))
        labels.append(row[7]) # the 8th column is the label
    gtFile.close()
    return (np.array(features, dtype='float32'), utils.to_categorical(labels, 43))     
