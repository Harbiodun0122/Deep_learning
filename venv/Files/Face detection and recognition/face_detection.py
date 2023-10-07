# We'll be using haarcascade classifier to detect faces present in an image
# harrcascade is so sensitive to noise in an image
# Local binary patterns can also be used for face detection
# They are both built in OpenCV package

from cv2 import cv2 as cv

img = cv.imread('../photos/IMG_20210726_075333.jpg')
# def rescaleFrame(frame, scale):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#
#     dimensions = width, height
#
#     return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
#
# resized_img = rescaleFrame(img, 0.20)
# cv.imshow('Myself', img)

# convert image to grayscale since face detection is not concerned about the colors present in the image
# haar cascade uses edges to detect faces
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray Me', gray)

# reading the haar_cascade.xml file
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# will use scaleFactor and minNeighbors to detect the face and return the rectangular coordinates of the face
# scaleFactor and minNeighbors can be tuned
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8)
print(f'No of face found = {len(faces_rect)}')

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# cv.imshow('Detected face', img)
#
# cv.waitKey(0)

# It can also be used on videos
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    gray_video = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if not isTrue:
        break

    video_faces_rect = haar_cascade.detectMultiScale(gray_video, scaleFactor=1.1, minNeighbors=5)
    print(f'No of face found = {len(video_faces_rect)}')

    for (x, y, w, h) in video_faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    cv.imshow('Face detectio', frame)
    # cv.imshow('Snapchat gray_video', gray_video)

    if cv.waitKey(30) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()