from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix,classification_report
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import random
import tensorflow as tf
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

numTestImages = 0

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
        numTestImages +=1

# print("Training array size", len(trainingImages))
# print("Training Label size", len(trainingLabels))
# print("Validation array size", len(validationImages))
# print("Validation array size", len(validationLabels))
# print("Testing array size", len(testImages))
# print("Testing array size", len(testLabels))

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

model.add(Dense(300, input_shape=(28*28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('tanh'))

model.add(Dense(300))
model.add(Activation('tanh'))

model.add(Dense(300))
model.add(Activation('tanh'))

#
# model.add(Dense(200))
# model.add(Activation('relu'))

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

NUM_EPOCHS = 20
BATCH_SIZE = 200

history = model.fit(trainingImages, trainingLabels,
                    validation_data = (validationImages, validationLabels),
                    epochs=NUM_EPOCHS,
                    verbose=1,
                    batch_size=BATCH_SIZE)



image_index = 35
print(history.history)

e = range(1,NUM_EPOCHS + 1)
plt.plot(e, history.history['accuracy'], color='blue', label = 'Training Accuracy')
plt.plot(e, history.history['val_accuracy'], color='red', label = 'Validation Accuracy')
plt.xticks(e)
plt.xlabel('Epochs')
plt.ylabel('%')
plt.title('Percent Accuracy per Epoch of Training')
plt.legend()
plt.show()

rounded_labels=np.argmax(testLabels, axis=1)

print("Shape of testing labels is ", testLabels.shape)
print("Shape of rounded labels is ", rounded_labels.shape)
print("Shape of testing images is ", testImages.shape)

test_predictions = model.predict(testImages)
confusion = confusion_matrix(rounded_labels, np.argmax(test_predictions,axis=1))

print(confusion)

pred_bool = np.argmax(test_predictions,axis=1)
print(classification_report(testLabels, pred_bool))