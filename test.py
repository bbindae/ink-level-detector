import cv2 as cv
img = cv.imread('./images/pic.png')
cv.imshow("Captured image", img)
cv.waitKey(0)
