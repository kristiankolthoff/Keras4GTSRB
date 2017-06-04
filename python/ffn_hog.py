import loader
import network_utils as nutils
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping
from keras.optimizers import Adadelta
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Data requirements
set = 'HOG_02'
#Model requirements
batchSize = 32
epochs = 13
#Load train and test data
print('Loading Training data...')
trainHOGFeatures, trainLabels = loader.readHOGFeaturesTrainData(set=set)
print('Loading Test data...')
testHOGFeatures, testLabels = loader.readHOGFeaturesTestData(set=set)

#Create input layer
input = Input(shape=trainHOGFeatures.shape[1:])
#First dense layer
dens1 = Dense(1000)(input)
act1  = Activation('relu')(dens1)
drop1 = Dropout(rate=0.5)(act1)
#Second dense layer
dens2 = Dense(1000)(drop1)
act2  = Activation('relu')(dens2)
drop2 = Dropout(rate=0.5)(act2)
#Third dense layer
dens3 = Dense(750)(drop2)
act3  = Activation('relu')(dens3)
drop3 = Dropout(rate=0.5)(act3)
#Fourth dense layer
dens4 = Dense(500)(act3)
act4  = Activation('relu')(dens4)
drop4 = Dropout(rate=0.5)(act4)
#Output layer 
dens5 = Dense(43)(drop4)
act5  = Activation('softmax')(dens5)

#Create model instance with input and outputs
model = Model(inputs=[input], outputs=[act5])
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
#Create model summary
model.summary()
model.fit(trainHOGFeatures, trainLabels, callbacks = [EarlyStopping(monitor = 'loss', 
                                                    min_delta = 0.03, patience = 1)],
          batch_size=batchSize, epochs=epochs)
model.save('../models/fnn_hog.h5')

score = model.evaluate(testHOGFeatures, testLabels)
prediction = model.predict(testHOGFeatures)
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