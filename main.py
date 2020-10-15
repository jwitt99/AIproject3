from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import numpy as np

# Model Template
# https://www.sitepoint.com/keras-digit-recognition-tutorial/

#Load in images array
npyImages = np.load("images.npy")
npyLabels = np.load('labels.npy')

i = npyImages
l = npyLabels

# image_index = 35
# print(npyLabels[image_index])
# plt.imshow(npyImages[image_index], cmap='Greys')
# plt.show()

trainingImages = []
trainingLabels = []
validationImages = []
validationLabels = []
testImages = []
testLabels = []

random.seed()

npyImages = npyImages.reshape(6500, 784)
for x in range(0, 6500):
    randVal = random.randint(0,100)
    if(randVal <= 60):
        trainingImages.append(npyImages[x])
        trainingLabels.append(npyLabels[x])
    elif(randVal > 60 and randVal <75):
        validationImages.append(npyImages[x])
        validationLabels.append(npyLabels[x])
    else:
        testImages.append(npyImages[x])
        testLabels.append(npyLabels[x])

# print("Training array size", len(trainingImages))
# print("Training Label size", len(trainingLabels))
# print("Validation array size", len(validationImages))
# print("Validation array size", len(validationLabels))
# print("Testing array size", len(testImages))
# print("Testing array size", len(testLabels))



trainingImages = np.asarray(trainingImages)
trainingImages = np.asarray(trainingImages)
trainingLabels = np.asarray(trainingLabels)
validationImages = np.asarray(validationImages)
validationLabels = np.asarray(validationLabels)
testImages = np.asarray(testImages)
testLabels = np.asarray(testLabels)


#divides data into three sets

imgRows = 28
imgCols = 28



print("training images shape after", trainingImages.shape)

numClasses = 10

trainingLabels = to_categorical(trainingLabels, numClasses)
validationLabels = to_categorical(validationLabels, numClasses)
testLabels = to_categorical(testLabels, numClasses)


#
# model.add(Conv2D(32, kernel_size=(3, 3),
#      activation='relu',
#      input_shape=(imgRows, imgCols, 1)))


model = Sequential()  # declare model

model.add(Dense(50, input_shape=(28*28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))

model.add(Dense(300))
model.add(Activation('relu'))

model.add(Dense(500))
model.add(Activation('relu'))

model.add(Dense(1000))
model.add(Activation('relu'))
#
#
# Fill in Model Here
#
#
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))
# # Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
# x_train = training images
# y_train = training labels
# x_val = validation images
# y_val = validation labels

print("Shape of training images is ", trainingImages.shape)
print("Shape of training labels is ", trainingLabels.shape)

history = model.fit(trainingImages, trainingLabels,
                    validation_data = (validationImages, validationLabels),
                    epochs=20,
                    verbose=1,
                    batch_size=512)



image_index = 35
print(history.history)

# plt.imshow(npyImages[image_index], cmap='Greys')
# plt.show()
#
# print(testLabels[image_index].index(1))
# gray = testImages[image_index]
# gray = gray/255
# predicts = model.predict(gray)
# print(predicts.argmax())