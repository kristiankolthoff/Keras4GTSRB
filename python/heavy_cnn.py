#Implements the architecure used in one of the
#competitors of the original challenge. 
# 
# Paper : Multi-Colum deep neural network
#         for traffic sign classification 
#         (Ciresan et al. 2012)
import loader
import network_utils as nutils
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.losses import categorical_crossentropy
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

#Create a sequential model to stack the
#layers linearly on top of each other
model = Sequential()
#First convolutional layer outputs tensor
#with shape (42,42,100)
model.add(Conv2D(100, kernel_size=(7,7),
                 input_shape=trainImages.shape[1:], data_format="channels_last"))
model.add(Activation('relu'))
#First maximum pooling layer outputs tensor
#with shape (21,21,100)
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(rate=0.25))
#Second convolutional layer outputs tensor
#with shape (18,18,150)
model.add(Conv2D(150, kernel_size=(4,4)))
model.add(Activation('relu'))
#Second maximum pooling layer outputs tensor
#with shape (9,9,150)
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(rate=0.25))
#Third convolutional layer outputs tensor
#with shape (6,6,250)
model.add(Conv2D(250, kernel_size=(4,4)))
model.add(Activation('relu'))
#Third maximum pooling layer outputs tensor
#with shape (3,3,250)
model.add(MaxPool2D(pool_size=(2,2)))
#model.add(Dropout(rate=0.25))
#Flatten the input tensor with shape of
#(3,3,250) to output tensor with shape (2250,)
model.add(Flatten())
#One fully-conntected idden layer with 300 neurons
model.add(Dense(300))
model.add(Activation('relu'))
#Dropout to avoid overfitting
model.add(Dropout(rate=0.5))
#Final output layer with 43 neurons and softmax
#activation function
model.add(Dense(43, activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
#Create model summary
model.summary()
model.fit(normTrainImages, trainLabels, batch_size=batchSize, epochs=epochs)
model.save('../models/heavy_cnn.h5')
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