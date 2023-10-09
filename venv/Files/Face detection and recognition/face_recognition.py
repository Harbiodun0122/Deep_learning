import os
import numpy as np
from cv2 import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Mbappe', 'Messi', 'Neymar', 'Pedri', 'Ronaldo', 'Suarez']
DIR = r'train'

features = np.load('features.npy', allow_pickle=True)
features = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

images = []
for person in people:
    path = os.path.join(DIR, person)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)

        img_array = cv.imread(img_path)
        images.append(img_array)

for image in images:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imshow('Person', gray)

    # detect the face in the image
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y + h, x:x + h]

        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with confidence {confidence}')

        cv.putText(image, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Detected Face', image)
    cv.waitKey(0)
