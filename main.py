from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
import random
import numpy as np

# Model Template

#Load in images array
npyImages = np.load("images.npy")

images = []
for i in npyImages:
    newArr = i.reshape(-1)
    images.append(newArr)

# Load in nmpy labels array and prints it
npyLabels = np.load('labels.npy')
print(npyLabels)

# one hot encode
labels = to_categorical(npyLabels)


#divides data into three sets

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
        trainingImages.append(images[x])
        trainingLabels.append(labels[x])
    elif(randVal > 60 and randVal <75):
        validationImages.append(images[x])
        validationLabels.append(labels[x])
    else:
        testImages.append(images[x])
        testLabels.append(labels[x])

print("Training array size", len(trainingImages))
print("Validation array size", len(validationImages))
print("Testing array size", len(testImages))
# ---------------------------------------------------

# You are given 6500 images and labels. The training set should
# contain ~60% of the data, the validation set should contain ~15%
# of the data, and the test set should contain ~25% of the data.

model = Sequential()  # declare modelnd
model.add(Dense(10, input_shape=(28 * 28,), kernel_initializer='he_normal'))  # first layer
model.add(Activation('relu'))
#
#
#
# Fill in Model Here
#
#
model.add(Dense(10, kernel_initializer='he_normal'))  # last layer
model.add(Activation('softmax'))

# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
# x_train = training images
# y_train = training labels
# x_val = validation images
# y_val = validation labels
# history = model.fit(x_train, y_train,
#                     validation_data = (x_val, y_val),
#                     epochs=10,
#                     batch_size=512)


# Report Results

# print(history.history)
# model.predict()
