import cv2
from matplotlib import pyplot as plt
import imutils

# read image and take first channel 
# only
bottle_3_channel = cv2.imread("./images/pen3.png")
bottle_gray = cv2.split(bottle_3_channel)[0]
cv2.imshow("Bottle Gray", bottle_gray)
cv2.waitKey(0)

# blur image
bottle_gray = cv2.GaussianBlur(bottle_gray, (7, 7), 0)
cv2.imshow("Bottle Gray smoothed 7 x 7", bottle_gray)
cv2.waitKey(0)

# draw histogram
plt.hist(bottle_gray.ravel(), 256, [0, 256])
plt.show()

# manual threshold
(T, bottle_threshold) = cv2.threshold(bottle_gray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Bottle Gray Threshold 27.5", bottle_threshold)
cv2.waitKey(0)

#apply opening operation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
bottle_open = cv2.morphologyEx(bottle_threshold, cv2.MORPH_OPEN, kernel)
cv2.imshow("Bottle Open 5 x 5", bottle_open)
cv2.waitKey(0)

# find all contours
contours = cv2.findContours(bottle_open.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
bottle_clone = bottle_3_channel.copy()
cv2.drawContours(bottle_clone, contours, -1, (255, 0, 0), 2)
cv2.imshow("All Contours", bottle_clone)
cv2.waitKey(0)

# sort contours by area
areas = [cv2.contourArea(contour) for contour in contours]
(contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
# print contour with largest area
bottle_clone = bottle_3_channel.copy()
cv2.drawContours(bottle_clone, [contours[-1]], -1, (255, 0, 0), 2)
cv2.imshow("Largest contour", bottle_clone)
cv2.waitKey(0)

# draw bounding box, calculate aspect and display decision
bottle_clone = bottle_3_channel.copy()
(x, y, w, h) = cv2.boundingRect(contours[-1])
aspectRatio = w / float(h)
if aspectRatio < 0.4:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(bottle_clone, "Full", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
else:
    cv2.rectangle(bottle_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(bottle_clone, "Low", (x + 10, y + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
cv2.imshow("Decision", bottle_clone)
cv2.waitKey(0)