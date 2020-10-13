from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

# Model Template

images = np.load("images.npy")

vecArr = []
for i in images:
    newArr = i.reshape(-1)
    vecArr.append(newArr)
print(vecArr[0].size)

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
# history = model.fit(x_train, y_train,
#                     validation_data = (x_val, y_val),
#                     epochs=10,
#                     batch_size=512)


# Report Results

# print(history.history)
# model.predict()
