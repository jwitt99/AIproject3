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

print("Training array size", len(trainingImages))
print("Training Label size", len(trainingLabels))
print("Validation array size", len(validationImages))
print("Validation array size", len(validationLabels))
print("Testing array size", len(testImages))
print("Testing array size", len(testLabels))

trainingImages = np.asarray(trainingImages)
trainingLabels = np.asarray(trainingLabels)
validationImages = np.asarray(validationImages)
validationLabels = np.asarray(validationLabels)
testImages = np.asarray(testImages)
testLabels = np.asarray(testLabels)

tempImage = trainingImages

#divides data into three sets
images = []
imgRows = 28
imgCols = 28

for i in trainingImages:
    newArr = i.reshape(-1)
    images.append(newArr)

trainingImages = np.asarray(images)

print("training images shape after", trainingImages.shape)

# trainingImages = trainingImages/255
# validationImages = validationImages/255
# testImages = testImages/255


numClasses = 10

trainingLabels = to_categorical(trainingLabels, numClasses)
validationLabels = to_categorical(validationLabels, numClasses)
testLabels = to_categorical(testLabels, numClasses)


#
# model.add(Conv2D(32, kernel_size=(3, 3),
#      activation='relu',
#      input_shape=(imgRows, imgCols, 1)))


model = Sequential()  # declare model

model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))
# model.add(Dense(30))
# model.add(Activation('relu'))
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
                    epochs=10,
                    verbose=1,
                    batch_size=512)



image_index = 35
print(history.history)

# plt.imshow(npyImages[image_index], cmap='Greys')
# plt.show()

print(npyLabels[image_index])
gray = npyImages[image_index].reshape(1, imgRows, imgCols, 1)
gray = gray/255
predicts = model.predict(gray)
print(predicts.argmax())