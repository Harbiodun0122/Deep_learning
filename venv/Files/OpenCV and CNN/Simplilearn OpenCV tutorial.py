# pip install opencv-python    ~ to install the main package
# pip install opencv-contrib-python-headless
from cv2 import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# reading images in opencv
image = cv.imread('../photos/IMG_20210908_122246.jpg')
'''cv.imshow('Image', image)
cv.waitKey(0)'''

# resizing images and videos
'''resized_image = cv.resize(image, (400, 533), -1)
cv.imshow('Resized Image', resized_image)
cv.waitKey(0)'''

# reading videos
'''capture = cv.VideoCapture('../videos/January.mp4')
while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break
    cv.imshow('Video', frame)

    if cv.waitKey(10) & 0xFF==ord('q'):
        break
capture.release()
cv.destroyAllWindows()
cv.waitKey(60)'''

# resizing image, this respects the aspect ratio of the image
'''def rescaleFrame(frame, scale):
    # Images, Video and Live Video
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = width, height

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def changeRes(width, height):
    # Live video
    capture.set(3, width)
    capture.set(4, height)'''

# Reading the resized video
'''while True:
    isTrue, frame = capture.read()

    if not isTrue:
        break
    frame_resized = rescaleFrame(frame, scale=.5)
    cv.imshow('Resized Video', frame_resized)

    if cv.waitKey(10) & 0xFF==ord('q'):
        break
capture.release()
cv.destroyAllWindows()
cv.waitKey(60)'''

# drawing with open cv
'''blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)
cv.waitKey(0)'''

# red box GBR format
'''blank[200:300, 300:400] = 0, 0, 255
cv.imshow('Blank', blank)
cv.waitKey(0)'''

# draw a green rectangle
'''cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=-1)
cv.imshow('Rectangle', blank)
cv.waitKey(0)'''

# draw a circle
'''cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 255, 0), thickness=-1)
cv.imshow('Circle', blank)
cv.waitKey(0)'''

# draw a line
'''cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)
cv.imshow('Line', blank)
cv.waitKey(0)'''

# write text
'''cv.putText(blank, 'Hello, my name is Abiodun!!!', (0, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
cv.imshow('Text', blank)
cv.waitKey(0)'''

# BASIC IMAGE FUNCTIONS

# pylint:disable=no-member
img = cv.imread('../photos/IMG_20210726_075333.jpg')
print(img.shape)
'''cv.imshow('Me', img)
cv.waitKey(0)'''

# converting to grayscale
'''gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
cv.waitKey(0)'''

# blur
# blur1 = cv.GaussianBlur(img, (0,0), cv.BORDER_DEFAULT)
'''blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
cv.waitKey(0)'''

# edge cascade
'''canny = cv.Canny(blur, 125, 175) # this is used for edge detection
canny2 = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)
cv.imshow('Image Canny Edges', canny2)
cv.waitKey(0)'''

# dilating the image
'''dilated = cv.dilate(canny, (7, 7), iterations=2) # dilate takes in the canny, iterations can be anything
cv.imshow('Dilated', dilated)
cv.waitKey(0)'''

# Eroding
'''eroded = cv.erode(dilated, (7, 7), iterations=3)
cv.imshow('Eroded', eroded)
cv.waitKey(0)'''

# resize
'''resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)
cv.waitKey(0)'''

# cropping
'''cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)
cv.waitKey(0)'''

# Image Transformation

# Translation # means shifting an image along the x and y axis
'''def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0]) # width, height
    return cv.warpAffine(img, transMat, dimensions)

translated = translate(img, -100, 100)
cv.imshow('Translated', translated)
cv.waitKey(0)'''

# rotation
'''def rotate(img, angle, rotPoint=None):
    height, width = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = width, height

    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)
cv.imshow('Rotated', rotated)
cv.waitKey(0)'''

# flipping
'''flip = cv.flip(img, 1)
cv.imshow('flip', flip)
cv.waitKey(0)'''

# CONTOUR DETECTION is a useful tool in shape analysis and object detection and recognition
'''blank = np.zeros(img.shape, dtype='uint8')
cv.imshow('Me', blank)
# cv.waitKey(0)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# cv.waitKey(0)

blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)
# # cv.waitKey(0)
#
canny = cv.Canny(blur, 125, 175)
cv.imshow('Canny Edges', canny)
# # cv.waitKey(0)

ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Thresh', thresh)
# # cv.waitKey(0)

contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contours(s) found!')

cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours drawn', blank)
cv.waitKey(0)'''

# color spaces
'''plt.imshow(img)
plt.show()
cv.waitKey(0)'''

# bgr to hsv
'''hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)
cv.waitKey(0)'''

# bgr to l*a*b
'''lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)
cv.waitKey(0)'''

# bgr to rgb
'''rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)
cv.waitKey(0)'''

# bgr to rgb
'''lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('lab --> bgr', lab_bgr)
cv.waitKey(0)'''

# Color channels
'''blank = np.zeros(img.shape[:2], dtype='uint8')
b, g, r = cv.split(img)
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g,  blank])
red = cv.merge([blank, blank, r])
cv.imshow('blue', blue)
# cv.waitKey(0)
cv.imshow('green', green)
# cv.waitKey(0)
cv.imshow('red', red)
# cv.waitKey(0)

merged = cv.merge([b, g, r])
cv.imshow('merged image', merged)
cv.waitKey(0)'''

# blur is used to reduce noise
# averaging ---> it blurs more than guassian blur
'''average = cv.blur(img, (5, 5))
cv.imshow('average blur', average)
cv.waitKey(0)'''

# gaussian blur
'''gauss = cv.GaussianBlur(img, (5,5), 0)
cv.imshow('gauss Blur', gauss)
cv.waitKey(0)'''

# median blur
'''median = cv.medianBlur(img, 5)
cv.imshow('Median Blur', median)
cv.waitKey(0)'''

# bilateral {most advanced form of blurring, it blurs and also retains the edges of the image}
'''bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral', bilateral)
cv.waitKey(0)'''

# BITWISE OPERATIONS
'''blank = np.zeros((400, 400), dtype='uint8')
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)
cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)
cv.waitKey(0)'''

# bitwise intersecting regions
'''bitwise_and = cv.bitwise_and(rectangle, circle) # intersecting regions
bitwise_or = cv.bitwise_or(rectangle, circle) # non-intersecting and intersecting regions
bitwise_not = cv.bitwise_not(rectangle, circle) # inverts the binary color
bitwise_xor = cv.bitwise_xor(rectangle, circle) # non-intersecting regions
cv.imshow('bitwise_and', bitwise_and)
cv.imshow('bitwise_or', bitwise_or)
cv.imshow('bitwise_not', bitwise_not)
cv.imshow('bitwise_xor', bitwise_xor)
cv.waitKey(0)'''

# masking --> allows us to focus on one part of the image. e.g focusing on the face of a person
'''blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('blank', blank)
# cv.waitKey(0)

circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)
cv.imshow('circle', circle)
# cv.waitKey(0)

rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
cv.imshow('rectangle', rectangle)
# cv.waitKey(0)

mask =cv.bitwise_and(circle, rectangle)
cv.imshow('mask', mask)
# cv.waitKey(0)

masked =cv.bitwise_and(img, img, mask=mask)
cv.imshow('masked', masked)
cv.waitKey(0)'''

# Histograms
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
'''cv.imshow('Gray', gray)
cv.waitKey(0)

# grayscale histogram
gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
# plt.show()'''

# color scale histogram
'''plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
# plt.show()'''

# THRESHOLDING
# Simple Threshold
'''threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Threshold', thresh)
# cv.waitKey(0)

# simple threshold inverse
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Threshold Inverse', thresh_inv)
# cv.waitKey(0)

# adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 9)
cv.imshow('Adaptive_thresh', adaptive_thresh)
cv.waitKey(0)'''

# Edge detection
# Sobel
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
combined_sobel = cv.bitwise_or(sobelx, sobely)
cv.imshow('sobelx', sobelx)
cv.imshow('sobely', sobely)
cv.imshow('combined_sobel', combined_sobel)
cv.waitKey(0)

# laplacian
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)
cv.waitKey(0)

# canny --> CHECK THE TOP