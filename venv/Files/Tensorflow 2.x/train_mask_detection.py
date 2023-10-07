# THE MODEL USED IN THIS PROJECT IS A FUNCTIONAL MODEL
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input

# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4  # this is already the default
EPOCHS = 20
BS = 32     # batch size
DIRECTORY = '../datasets/face_mask'

data = []
labels = []

for category in os.listdir(DIRECTORY):
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224)) # Loads an image into PIL format.
        image = img_to_array(image) # Converts a PIL Image instance to a Numpy array
        image = preprocess_input(image) # Preprocesses a tensor or Numpy array encoding a batch of images

        data.append(image)
        labels.append(category)

print(data[:10])
print(labels[:10])

# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels) # instead of using onehotencoder
print('lb.classes_: ', lb.classes_)

# turning the data and labels list into a numpy array
data = np.array(data, dtype='float32')
labels = np.array(labels)
print(labels[-10:])

# splitting the dataset into training and testing set
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
print('Data Augmentation: Generating Image...')
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
                         horizontal_flip=True, fill_mode='nearest')
print('Finished Generating Image')

# load the mobilenetv2 network, ensuring the head FC layer sets are left off
print('Loading MobileNetV2 Neural Network...')
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
print('Finished Loading MobileNetV2 Neural Network')

# construct the head of the model that will be placed on top of the base model
headModel = baseModel.output # Retrieves the output tensor(s) of a layer.
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# place the head fc model on top of the base model (this will become the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will not be update during the first training process
# mostly done in series data like the stock market
for layer in baseModel.layers:
    layer.trainable = False 

# compile our model
print('[INFO] compiling model')
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# train the head of the network
print('[INFO] training head...')
H = model.fit(aug.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS, validation_data=(testX, testY),
              validation_steps=len(testX) // BS, epochs=EPOCHS)
print('H.history loss: \n', H.history['loss'])
print('H.history accuracy: \n', H.history['accuracy'])
print('H.history val_loss: \n', H.history['val_loss'])
print('H.history val_accuracy: \n', H.history['val_accuracy'])

# make predictions on the testing set
print('[INFO] evaluating network....')
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set, we need to find the index of the label with the corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
print('predIdxs: ', predIdxs)
print('testY: ', testY.argmax(axis=1))

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk
print('[INFO]  saving mask detector model...')
model.save('mask_detector.model', save_format='h5')

# plt the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, EPOCHS), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, EPOCHS), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, EPOCHS), H.history['val_accuracy'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='center right')
plt.savefig('plot.png')
plt.show()