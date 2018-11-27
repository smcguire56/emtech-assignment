# Initially import all libraries and associted methods with them
import gzip
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Dropout, Flatten  
from keras.datasets import mnist
import sklearn.preprocessing as pre

# Read in the training data images and the labels
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
    train_img = f.read()

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_lbl = f.read()


with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_img = f.read()

with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_lbl = f.read()

# show that the data is currently in byte format
print(type(train_img))
print(type(train_lbl))

# First Image
train_img[16:800]

# convert from bytes to integers in big endian format.
#src: https://www.webopedia.com/TERM/B/big_endian.html & https://stackoverflow.com/questions/846038/convert-a-python-int-into-a-big-endian-string-of-bytes
int.from_bytes(train_img[16:800], 'big')

# input image dimensions
img_rows, img_cols = 28, 28

# Initialise a sequential keras model, sequentially adding layers.
# src: https://keras.io/getting-started/sequential-model-guide/
# single input, single output: sequential
model = Sequential()

# Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Add a three neuron output layer.
# ranges from 1 to 10
model.add(Dense(units=10, activation='softmax'))

print(model.summary())

# Build the graph.
model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])

train_img = ~np.array(list(train_img[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0
test_img = ~np.array(list(test_img[16:])).reshape(10000,28,28,1).astype(np.uint8) / 255.0
train_lbl =  np.array(list(train_lbl[ 8:])).astype(np.uint8)
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

x_train = train_img.reshape(60000,28,28,1)
y_train = keras.utils.to_categorical(train_lbl, 10)
x_test = test_img.reshape(10000,28,28,1)
y_test = keras.utils.to_categorical(test_lbl, 10)

encoder = pre.LabelBinarizer()
encoder.fit(train_lbl)
outputs = encoder.transform(train_lbl)

print(train_lbl[0], outputs[0])

for i in range(10):
    print(i, encoder.transform([i]))

model.fit(x_train, y_train,
          batch_size=10000,
          epochs=5,
          verbose=1)

(encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()

model.predict(test_img[5:6])

plt.imshow(test_img[5].reshape(28, 28), cmap='gray')
plt.savefig('fig')
