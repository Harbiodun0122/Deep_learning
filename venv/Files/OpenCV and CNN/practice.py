import os
import numpy as np
from cv2 import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

DIRECTORY = r'C:\Users\Harbiodun\Documents\Data Science and AI\Deep Learning\venv\Files\datasets\single_prediction'

print('loading model...')
maskNet = load_model('cat-and-dog25.model')
print('model loading complete')

IMG = []

def grab_from_directory(directory):
    for images in os.listdir(directory):
        image = os.path.join(directory, images)
        # show_img = cv2.imread(image)
        # cv2.imshow('Image', show_img)
        # cv2.waitKey(0)
        image = load_img(image, target_size=(64, 64))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        IMG.append(image)
    return IMG

def make_prediction(img):
    '''This function handles the prediction made by the model
    It takes an argument 'img' which the the output of the grab_from_directory function'''
    print('making predictions...')
    animals = []
    for images in img:
        result = maskNet.predict([images])
        print(result)
        if result[0][0] == 1:
            prediction = 'dog'
            animals.append(prediction)
        elif result[0][0] == 0:
            prediction = 'cat'
            animals.append(prediction)
        else:
            prediction = 'unknown'
            animals.append(prediction)
    return animals

image = grab_from_directory(DIRECTORY)
final_result = make_prediction(image)
print(final_result)