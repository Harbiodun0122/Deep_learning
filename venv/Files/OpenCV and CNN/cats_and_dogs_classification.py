import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

# we'll just go straigt to the cnn because no feature engineering, data cleaning e.t.c has to be done on the data

# initialising the CNN, Sequential is the actual model/neural network we are using
classifier = Sequential()

# INPUT LAYERS
# convolution layer ---> 1st layer, it does the most work
# Conv2D converts the 3D photo to 2D
# input_shape must match the shape of data coming in
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
####################
classifier.add(Dropout(0.25))
###########################

# HIDDEN LAYERS
# adding a second convolutional layer
# MaxPooling2D is doing the mapping and reducing(MapReduce) and reducing it to two sets (2,2) size
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
####################
classifier.add(Dropout(0.25))
###########################

###########################
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
classifier.add(Dropout(0.25))
##########################

# flattening ---> this is very important
# just for it to work with a single array and not multiple dimensions of data
classifier.add(Flatten())

# full connection
# the Reduce part of MapReduce
classifier.add(Dense(units=128, activation='relu'))
##########################
# classifier.add(Dense(units=64, activation='relu'))
##########################
# units=1 makes it single output. Sigmoid makes it clear that we only want two outputs since the units has been set to 1
classifier.add(Dense(units=1, activation='sigmoid'))

# compiling the CNN
# optimizer is the reverse propagation
# adam works best on large amount of data
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Analyzing model:
print(classifier.summary())

# fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
###############################################################
training_set = train_datagen.flow_from_directory(
    r'C:\Users\Harbiodun\Documents\Data Science and AI\Deep Learning\venv\Files\datasets\dog vs cat\dataset\training_set',
    target_size=(64, 64), batch_size=32,
    class_mode='binary')
test_set = test_datagen.flow_from_directory(
    r'C:\Users\Harbiodun\Documents\Data Science and AI\Deep Learning\venv\Files\datasets\dog vs cat\dataset\test_set',
    target_size=(64, 64), batch_size=32,
    class_mode='binary')
###############################################################
# fit_generator is the back propagation
# classifier.fit(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000) # for commercial release
EPOCH = 10
H = classifier.fit(training_set, steps_per_epoch=8000 // 32, epochs=EPOCH, validation_data=test_set,
                   validation_steps=2000 // 32)
classifier.save('cat-and-dog-10.model', save_format='h5')

plt.figure()
plt.plot(np.arange(0, EPOCH), H.history['loss'], label='loss')
plt.plot(np.arange(0, EPOCH), H.history['accuracy'], label='accuarcy')
plt.plot(np.arange(0, EPOCH), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, EPOCH), H.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy and loss')
plt.title('Training loss and accuracy')
plt.legend(loc='center right')
plt.savefig('cats_and_dog-10.png')
plt.show()
