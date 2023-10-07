# Importing OpenCV, time and Pandas Library
from cv2 import cv2
import time, pandas
# importing datetime class from datetime library
from datetime import datetime
# assigning our static_back to None
static_back = None

# initialising dataframe, one column is start time and other column is end time
currentMotion = ['', '']
motionList = []

# capturing video
video = cv2.VideoCapture(0)

# infinite while loop to treat stack of image as video

while True:
    # Reading frame(image) from video
    check, frame = video.read()

    # initialising motion = 0(no motion)
    motion = 0

    # converting color image to grayscale image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # converting grayscale image to GuassianBlur so that change can be found easily
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # in first iteration we assign the value of static_back to our first frame
    if static_back is None:
        static_back = gray
        continue

    # Difference between static background and current frame(which is GuassianBlur)
    diff_frame = cv2.absdiff(static_back, gray)

    # if change in between static background ans current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # finding contour of moving objects
    cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            motion = 1
            continue
        if currentMotion[0] == '' and motion == 1:
            currentMotion[0] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        x, y, w, h = cv2.boundingRect(contour)
        # making green rectangle aroung the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Appending status of motion
    if motion == 0 and currentMotion[0] != '':
        currentMotion[1] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        motionList.append(currentMotion)
        currentMotion = ['', '']

    # Displaying image in gray_scale
    cv2.imshow('Gray Frame', gray)

    # Displaying the difference in currentframe to the staticframe(very first_time)
    cv2.imshow('Difference Frame', diff_frame)

    # Displaying the black and white image in which if intensity difference greater than 30 it will appear white
    cv2.imshow('thresh Frame', thresh_frame)

    # Displaying color frame with contour of motion of object
    cv2.imshow('Color Frame', frame)

    key = cv2.waitKey(1)
    # if q entire process will stop
    if key == ord('q'):
        # if something is moving, then it append the end time of movement
        if motion == 1 and currentMotion[0] != '':
            currentMotion[1] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            motionList.append(currentMotion)
        break

for move in motionList[:30]:
    print(move)

video.release()

# destroying all the windows
cv2.destroyAllWindows()