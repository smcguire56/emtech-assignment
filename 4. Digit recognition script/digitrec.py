import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# The Convolutional Neural Networks expects a 4D array
# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train/=255
X_test/=255

# one hot encoding for labels, the only output is a number between 1 and 10 
# eg 5 : [0,0,0,0,0,1,0,0,0,0]

number_of_classes = 10
epochs = 5
batch_size=200

y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

# create Convolutional model
#  http://cs231n.github.io/convolutional-networks/

model = Sequential()
# first layer: 32 filters/ output channels, of size 5 x 5.  input layer expects image of structure height, width and channels
model.add(Conv2D(32, (5, 5), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation='relu'))
# max pooling layer, reduces the over-fitting
model.add(MaxPooling2D(pool_size=(2, 2)))
# another hidden layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# last layer, 10 neorons, gives probability of the class, binary classification of specified number.
model.add(Dense(number_of_classes, activation='softmax'))

# prints layer types, the shape of output and parameters. 
print(model.summary())

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

metrics = model.evaluate(X_test, y_test, verbose=0)

print(metrics)