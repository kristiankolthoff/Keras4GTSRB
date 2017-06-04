import loader
import network_utils as nutils
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Dense
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
fraction_training = 1
fraction_testing = 1
normalization = "adahistequi_cv2_rgb"
use_grayscale = False
#Model requirements
batchSize = 32
epochs = 20
#Load train and test data
print("Load training data...")
trainImages, trainLabels = loader.readTrainData(targetWidth, 
                                                targetHeight, classes, 
                                                fraction=fraction_training, 
                                                shuff=True,
                                                normalization = normalization,
                                                grayscale = use_grayscale)
#Normalize training image values
normTrainImages = trainImages/255
print("Load test data...")
testImages, testLabels = loader.readTestData(targetWidth,
                                             targetHeight,
                                             normalization = normalization,
                                             grayscale = use_grayscale,
                                             fraction=fraction_testing)
#Normalize testing image values
normTestImages = testImages/255


#Create sequenntial model type to stack up
#linear layers of convolution and pooling
model = Sequential()
#First convolution, pool pair layer
model.add(Conv2D(32, kernel_size=(3,3), input_shape=trainImages.shape[1:], activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.1))
#Second convolution, pool layer
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.1))
##Flatten the (8,8,64) output tensor to 
#tensor with shape (4096,)
model.add(Flatten())
#One fully-connected hidden layer with
#dropout of 0.5 applied
model.add(Dense(128))
model.add(Dropout(rate=0.5))
#Output classification layer with softmax
#activation function
model.add(Dense(43, activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
#Summarize comppiled architecture
model.summary()
model.fit(normTrainImages, trainLabels, batch_size=batchSize, epochs=epochs)
#Save model architecture and learned weights to file to be used later on
model.save("../models/light_cnn_" + normalization + ".h5")
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