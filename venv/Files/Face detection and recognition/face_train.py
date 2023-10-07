# We are using OpenCV built-in face recognizer
import os
import numpy as np
from cv2 import cv2 as cv

people = people = ['Mbappe', 'Messi', 'Neymar', 'Pedri', 'Ronaldo', 'Suarez']
# alternative way
# p = []
# for i in os.listdir(DIR):
#     p.append(i)
# print(p)

DIR = r'C:\Users\USER\Documents\Data Science and AI\Deep Learning\venv\Files\Face detection and recognition\train'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = [] # face of the person
labels = [] # the person the face belong to

# loop through every folder in the directory specified, grab the faces in the image and add it to the training set
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training complete................')
# print(f'Length of features = {len(features)}')
# print(f'Length of labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

# train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')

# save the training model
np.save('features.npy', features)
np.save('labels.npy', labels)