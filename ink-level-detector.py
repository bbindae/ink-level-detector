import cv2 as cv
from matplotlib import pyplot as plt
import imutils
from time import sleep


import RPi.GPIO as GPIO
from picamera import PiCamera



def capture_image(cv, width, height, show_image):
    # Set full HD resolution
    camera = PiCamera(resolution=(1080, 1920), framerate=30)
    img = None
    #try:
    camera.start_preview()
    sleep(2)
    camera.stop_preview()
    camera.capture('./images/pic.png', resize=(width,height))
    camera.close()
    print("an image captured!")
    print("start reading an image captured")
    img = cv.imread('./images/pic.png')
    print("image read")
    if show_image:
        cv.imshow("Captured image", img)
        cv.waitKey(0)        
    #finally:
        #camera.close()
        #pass
    print("reading image done")
    return img    

def convert_to_gray_image(cv, img, show_image=True):
    print("convert to gray image")
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if show_image:
        cv.imshow("GrayScale image", gray_img)
        cv.waitKey(0)
    
    return cv.split(gray_img)[0]

def blur_image(cv, img, show_image=True):
    print("blue image")
    img = cv.GaussianBlur(img, (7, 7), 0)
    if show_image:
        cv.imshow("Gray smoothed 7 x 7", img)
        cv.waitKey(0)

    return img

def get_threshold(cv, plt, img, show_image):
    print("get threshold")
    if show_image:
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()
    
    (T, segmented_img) = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    #(T, segmented_img) = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)
    if show_image:
        cv.imshow("segmented image", segmented_img)
        print("Threshold: {}".format(T))
        cv.waitKey(0)
    
    return segmented_img

def apply_opening(cv, img, show_image):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open_image = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    if show_image:
        cv.imshow("Apply Opening Operation", open_image)
        cv.waitKey(0)
    
    return open_image

def find_contours(cv, original_img, img, show_image):
    contours = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #print("finding contours:")
    #print(contours)

    # applying imutils.grab_contours makes countours openCV version agnostic
    contours = imutils.grab_contours(contours)
    original_img_clone = original_img.copy()
    cv.drawContours(original_img_clone, contours, -1, (255, 0, 0), 2)
    if show_image == True:
        cv.imshow("All contours", original_img_clone)
        cv.waitKey(0)
    
    return contours

def find_largest_contours(cv, original_img, contours, show_image):
    areas = [cv.contourArea(contour) for contour in contours]
    #print("area:")
    #print(areas)
    (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
    #print("(contours, areas):")
    #print((contours, areas))
    if show_image:
        original_img_clone = original_img.copy()
        cv.drawContours(original_img_clone, [contours[-1]], -1, (255, 0, 0), 2)
        cv.imshow("Largest contour", original_img_clone)
        cv.waitKey(0)
    
    return (contours, areas)

def last_decision(cv, original_img, contours, show_image):
    # draw bounding box, calculate aspect and display decision
    original_img_clone = original_img.copy()
    (x, y, w, h) = cv.boundingRect(contours[-1])
    print(x,y,w,h)
    aspect_ratio = w / float(h)
    print("aspect ratio" + str(aspect_ratio))
    limit = 0.17

    if aspect_ratio < limit:
        cv.rectangle(original_img_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(original_img_clone, "OK", (x + 10, y + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
    else:
        cv.rectangle(original_img_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(original_img_clone, "NOT OK", (x + 10, y + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
    
    if show_image:
        cv.imshow("Decision", original_img_clone)
        cv.waitKey(0)



button = 12
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
print("starting....")
try:
    while True:
        if GPIO.input(button) == GPIO.HIGH:
            print("start capturing....")
            #capture_image(cv,540, 960, True)
            #sleep(2)
            original_img = capture_image(cv, 540, 960, True)
            img = convert_to_gray_image(cv, original_img, True)
            img = blur_image(cv, img, True)
            img = get_threshold(cv, plt, img, True)
            img = apply_opening(cv, img, True)
            contours = find_contours(cv, original_img, img, True)
            (contours, areas) = find_largest_contours(cv, original_img, contours, True)
            last_decision(cv, original_img, contours, True)

finally:
    GPIO.cleanup()

# original_img = capture_image(cv, 540, 960, False)
# img = convert_to_gray_image(cv, original_img, False)
# img = blur_image(cv, img, False)
# img = get_threshold(cv, plt, img, False)
# img = apply_opening(cv, img, False)
# contours = find_contours(cv, original_img, img, False)
# (contours, areas) = find_largest_contours(cv, original_img, contours, True)
# last_decision(cv, original_img, contours, True)
