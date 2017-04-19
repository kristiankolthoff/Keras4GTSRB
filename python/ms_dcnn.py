import loader
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta

#input data requirements
targetWidth = 48
targetHeight = 48
classes = 43
fraction_training = 1
fraction_testing = 1
#Model requirements
batchSize = 32
epochs = 12
#Load train and test data
trainImages, trainLabels = loader.readTrainData(targetWidth, 
                                                targetHeight, classes, 
                                                fraction=fraction_training, 
                                                shuff=True)
normTrainImages = trainImages/255
testImages, testLabels = loader.readTestData(targetWidth,
                                             targetHeight,
                                             fraction=fraction_testing)

normTestImages = testImages/255
train_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


train_gen_trans = ImageDataGenerator(
        featurewise_center=True,
        width_shift_range=0.05,
        height_shift_range=0.05)

model = Sequential()
model.add(Conv2D(100, kernel_size=(7,7), activation='relu',
                 input_shape=(targetWidth, targetHeight, 3), data_format="channels_last"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(150, kernel_size=(4,4), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(250, kernel_size=(4,4), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(300))
model.add(Dense(43, activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
model.summary()
model.fit(normTrainImages, trainLabels, batch_size=batchSize, epochs=epochs)
#model.fit_generator(train_gen.flow(trainImages, trainLabels, batch_size=batchSize),
                   # steps_per_epoch=len(trainImages), epochs=epochs)
score = model.evaluate(normTestImages, testLabels)
print(score)
model.save_weights('ms_dcnn')
