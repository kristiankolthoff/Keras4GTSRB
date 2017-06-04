#Implements the architecure used in one of the
#competitors of the original challenge using a
#more heavyweight CNN architecture. 
# 
# Paper : Traffic Sign Recognition with Multi-Scale
#         Convolutional Networks
#         (Sermanet et al. 2011)

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
from keras.layers import concatenate
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
fraction_training = 1
fraction_testing = 1
#Model requirements
batchSize = 32
epochs = 20
#Load train and test data
trainImages, trainLabels = loader.readTrainData(targetWidth, 
                                                targetHeight, classes, 
                                                fraction=fraction_training,
                                                normalization = normalization,
                                                grayscale = use_grayscale,
                                                shuff=True)
#Normalize training images
normTrainImages = trainImages/255
testImages, testLabels = loader.readTestData(targetWidth,
                                             targetHeight,
                                             grayscale = use_grayscale,
                                             normalization = normalization,
                                             fraction=fraction_testing)

#Normalize testing images
normTestImages = testImages/255

#Create input layer
input = Input(shape=trainImages.shape[1:])
#First convolutional layer outputs tensor
#with shape (42,42,100)
#First convolution, pool pair layer
conv1 = Conv2D(108, kernel_size=(3,3))(input)
act1  = Activation('relu')(conv1)
max1  = MaxPool2D(pool_size=(2,2))(act1)
drop1 = Dropout(rate=0.1)(max1)
#Second convolution, pool layer
conv2 = Conv2D(150, kernel_size=(3,3))(drop1)
act2  = Activation('relu')(conv2)
max2  = MaxPool2D(pool_size=(2,2))(act2)
drop2 = Dropout(rate=0.1)(max2)
##Flatten the (8,8,64) output tensor to 
#tensor with shape (4096,)
flat1  = Flatten()(drop2)
flat2  = Flatten()(drop1)
concat = concatenate([flat1, flat2])
#One fully-connected hidden layer with
#dropout of 0.5 applied
dens1 = Dense(100)(concat)
act3  = Activation('relu')(dens1)
drop3 = Dropout(rate=0.5)(act3)
#One fully-connected hidden layer with
#dropout of 0.5 applied
dens2 = Dense(50)(drop3)
act4  = Activation('relu')(dens2)
drop4 = Dropout(rate=0.5)(act4)
#Output classification layer with softmax
#activation function
dens3 = Dense(43)(drop4)
act4 = Activation('softmax')(dens3)
#Create model instance with input and outputs
model = Model(inputs=[input], outputs=[act4])
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(lr=1),
              metrics=['accuracy'])
#Create model summary
model.summary()
model.fit(normTrainImages, trainLabels, 
          callbacks = 
          [EarlyStopping(monitor = 'loss', min_delta = 0.01, patience = 1)],
          batch_size=batchSize, epochs=epochs)
model.save("../models/heavy_mscnn" + normalization + ".h5")
score = model.evaluate(normTestImages, testLabels)
prediction = model.predict(normTestImages)
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