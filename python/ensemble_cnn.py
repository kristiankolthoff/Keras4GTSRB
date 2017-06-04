import loader
import network_utils as nutils
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import average
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#input data requirements
targetWidth = 48
targetHeight = 48
classes = 43
use_grayscale = False
normalization = "histequi"
set = 'HOG_02'
fraction_training = 1
fraction_testing = 1
#Model requirements
batchSize = 32
epochs = 20
#Load train and test data
print('Loading Training Images...')
trainImages, trainLabels = loader.readTrainData(targetWidth, 
                                                targetHeight, classes, 
                                                fraction=fraction_training,
                                                normalization = normalization,
                                                grayscale = use_grayscale,
                                                shuff=True)
#Normalize training images
normTrainImages = trainImages/255
print('Loading Test Images...')
testImages, testLabels = loader.readTestData(targetWidth,
                                             targetHeight,
                                             grayscale = use_grayscale,
                                             normalization = normalization,
                                             fraction=fraction_testing)

#Normalize testing images
normTestImages = testImages/255

#Create input layer
print('Loading Training Data hog...')
trainHOGFeatures, trainLabels = loader.readHOGFeaturesTrainData(set=set)
print('Loading Test dat hog...')
testHOGFeatures, testLabels = loader.readHOGFeaturesTestData(set=set)

#Create FeedForward Network for HOG Features
#Create input layer
inpu1 = Input(shape=trainHOGFeatures.shape[1:])
#First dense layer
dens1 = Dense(100)(inpu1)
act1  = Activation('relu')(dens1)
drop1 = Dropout(rate=0.5)(act1)
#Second dense layer
dens2 = Dense(500)(drop1)
act2  = Activation('relu')(dens2)
drop2 = Dropout(rate=0.5)(act2)
#Third dense layer
dens3 = Dense(250)(drop2)
act3  = Activation('relu')(dens3)
drop3 = Dropout(rate=0.5)(act3)
#Output layer 
dens3 = Dense(43)(drop3)
act4  = Activation('softmax')(dens3)

#First convolutional layer outputs tensor
#with shape (42,42,100)
#First convolution, pool pair layer
inpu2 = Input(shape=trainImages.shape[1:])
conv1 = Conv2D(32, kernel_size=(3,3))(inpu2)
act5  = Activation('relu')(conv1)
max1  = MaxPool2D(pool_size=(2,2))(act5)
drop4 = Dropout(rate=0.1)(max1)
#Second convolution, pool layer
conv2 = Conv2D(64, kernel_size=(3,3))(drop4)
act6  = Activation('relu')(conv2)
max2  = MaxPool2D(pool_size=(2,2))(act6)
drop5 = Dropout(rate=0.1)(max2)
##Flatten the (8,8,64) output tensor to 
#tensor with shape (4096,)
flat1  = Flatten()(drop5)
#One fully-connected hidden layer with
#dropout of 0.5 applied
dens4 = Dense(100)(flat1)
drop6 = Dropout(rate=0.5)(dens4)
act7  = Activation('relu')(drop6)
#One fully-connected hidden layer with
#dropout of 0.5 applied
dens5 = Dense(50)(act7)
drop7 = Dropout(rate=0.5)(dens5)
#Output classification layer with softmax
#activation function
dens7 = Dense(43)(drop7)
act8  = Activation('softmax')(dens7)
ave   = average([act4, act8])
#Create model instance with input and outputs
model = Model(inputs=[inpu1, inpu2], outputs=[ave])
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
#Create model summary
model.summary()
model.fit([trainHOGFeatures, normTrainImages], trainLabels,
          callbacks = 
          [EarlyStopping(monitor = 'loss', min_delta = 0.01, patience = 1)],
          batch_size=batchSize, epochs=epochs)
model.save('../models/ensemble_cnn_ffnn.h5')
score = model.evaluate([testHOGFeatures, normTestImages], testLabels)
prediction = model.predict([testHOGFeatures, normTestImages])
prediction = nutils.fromCategorical(prediction)
#Evaluation
testLabelsClasses = nutils.fromCategorical(testLabels)
cm = confusion_matrix(testLabelsClasses, prediction)
report = classification_report(testLabelsClasses, prediction)
f1Scores = nutils.get_f1_per_group(testLabelsClasses, prediction)
acc = accuracy_score(testLabelsClasses, prediction)
print(report)
print(score)
print(f1Scores)