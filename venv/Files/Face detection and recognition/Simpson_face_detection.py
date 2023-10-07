import gc
import os
import caer
import camaro
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2 as cv
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler


# computer vison model expects all image data size should be the same
IMG_SIZE = (80, 80)
channels = 1
char_path = 'characters path'  # base path where the images are stored in

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# sort in descending order
char_dict = caer.sort_dict(char_dict, descending=True)

characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break

# create the training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)
print(len(train))

# visualize the data
plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap='gray')
# plt.show()

# separate
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# normalize the feature set to be in the range of 0, 1 for the network to work on the data more faster
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))

# create training and validation data
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=.2)

# to save space --> i think this is a jupyter notebook thing
del train
del featureSet
del labels
gc.collect()

# create image data generator
BATCH_SIZE = 32
EPOCHS = 10

datagen = camaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# creating the model
model = camaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters),loss='Binary_crossentropy',
                                          decay=1e-6, learning_rate=0.001, momentum=0.9, nesterov=True)
print(model.summary)

# create a call back list to schedule learning rate so that the network can train better
callbacks_list = [LearningRateScheduler(camaro.lr_schedule)]

# train the model
training = model.fit(train_gen, steps_per_epoch=len(x_train)//BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val),
                     validation_steps = len(y_val)//BATCH_SIZE, callbacks=callbacks_list)

# use OpenCV to read in our image to see if the model actually performs well as said
img = cv.imread('file_path')

# resize the input image to the size and dimensions used for testing
def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return  img

predictions = model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])