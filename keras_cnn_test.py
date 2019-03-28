"""
Reference:
    'https://machinelearningmastery.com/how-to-evaluate-pixel-scaling-methods-for-image-classification/'
"""

from keras import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical

"""
Description:
    MNIST Database of handwritten digits
    Train Dataset = 60,000 Images
                    28 * 28 Resolution
                    Greyscale Images : 0-255 Intensity Range
                    Classes = 10 i.e. (0,1,2,3,4,5,6,7,8,9)
    Test Dataset = 10,000 Images
"""


# Loading the data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# summarize dataset shape
print('Train', train_images.shape, train_labels.shape)
print('Test', test_images.shape, test_labels.shape)

# summarize pixel values
print('Train', train_images.min(), train_images.max(), train_images.mean(), train_images.std())
print('Test', test_images.min(), test_images.max(), test_images.mean(), test_images.std())

# reshape dataset to have a channel indicator, in our
# case it would be 1 as our images are greyscale
width, height, channels = train_images.shape[1], train_images.shape[2], 1
train_images = train_images.reshape((train_images.shape[0], width, height, channels))
test_images = test_images.reshape((test_images.shape[0], width, height, channels))

# normalization  of pixel intensities
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# encoding of labels / target values
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# define model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(width, height, channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# evaluate model
_, acc = model.evaluate(test_images, test_labels, verbose=0)
print(acc)
