import cv2 as cv
from matplotlib import pyplot as plt
import imutils
from time import sleep






        
    

def read_captured_image(cv, file_path = '', show_image = True):
    print("# Step 2: Reading a captured image")
    path =  './images/pic.png' if file_path == '' else file_path
    print("\tStart reading a captured image")
    img = cv.imread(path)
    print("\tFinish reading a captured image")
    if show_image:
        show_image_center(cv, "Captured image", img)       
    
    return img  


def convert_to_gray_image(cv, img, show_image = True):
    print("\n# Step 3: Converting a captured image to a grayscale image")
    print("\tStart converting a captured image to a grayscale image...")
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("\tFinish converting a captured image to a grayscale image")
    if show_image:
        show_image_center(cv, "Grayscale image", gray_img)       

    return cv.split(gray_img)[0]

def blur_image(cv, img, show_image=True):
    print("\n# Step 4: Blurring a grayscale image using Gaussian smoothing")
    print("\tStart Gaussian smoothing...")
    img = cv.GaussianBlur(img, (7, 7), 0)
    print("\tFinish Gaussian smooting")
    if show_image:
        show_image_center(cv, "Gray smooted 7 x 7", img)
        
    return img

def get_threshold(cv, plt, img, show_image=True):
    print("\n# Step 5: Getting threshold for a binary image conversion")    
    
    if show_image:
        print("\tShowing a plotted histogram to help identify threshold")
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()
    
    (threshold, segmented_img) = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # (threshold, segmented_img) = cv.threshold(img, 50, 255, cv.THRESH_BINARY_INV)
    print("\tThreshold: {}".format(threshold))
        
    if show_image:
        show_image_center(cv, "Binary image", segmented_img)        
    
    return segmented_img

def apply_opening(cv, img, show_image=True):
    print("\n# Step 6: Applying morphological opening to clean up an image")
    print("\tStart eroding the foreground image...")
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open_image = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    print("\tDilating to restore the larger forground objects")
    if show_image:
        show_image_center(cv, "Apply Opening Operation", open_image)        
    
    return open_image

def find_contours(cv, original_img, img, show_image=True):
    print("\n# Step 7: Finding all contours of ink")
    print("\tStart finding all contours...")
    contours = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("\tFinished finding all contours")
    #print(contours)

    # applying imutils.grab_contours makes countours openCV version agnostic
    contours = imutils.grab_contours(contours)
    original_img_clone = original_img.copy()
    
    if show_image:
        cv.drawContours(original_img_clone, contours, -1, (255, 0, 0), 2)
        show_image_center(cv, "All contours", original_img_clone)
    
    return contours

def find_largest_contours(cv, original_img, contours, show_image=True):
    print("\n# Step 8: Finding the largest ink area")
    areas = [cv.contourArea(contour) for contour in contours]
    print("\tThe largest area: {}".format(areas))
    print("\tSorting a pair of contour and area by the size of area")
    (contours, areas) = zip(*sorted(zip(contours, areas), key=lambda a:a[1]))
    
    if show_image:
        original_img_clone = original_img.copy()
        cv.drawContours(original_img_clone, [contours[-1]], -1, (255, 0, 0), 2)
        show_image_center(cv, "The largest contour", original_img_clone)       

    return (contours, areas)

def show_decision(cv, original_img, contours, ink_level_threshold, show_image=True):
    print("\n# Step 9: Show a message with a green rectangle if the level of ink is enough.")
    print(" Otherwise, show a warning message with a red rectagle.")
   
    # draw bounding box, calculate aspect and display decision
    original_img_clone = original_img.copy()
    print("\tCalculating rectangle boundary")
    (x, y, w, h) = cv.boundingRect(contours[-1])
    print("\tX:{}, Y: {}, Width: {}, Height: {}".format(x, y, w, h))    
    aspect_ratio = w / float(h)
    print("\tAspect ratio: {}".format(aspect_ratio))    

    if aspect_ratio < ink_level_threshold:
        cv.rectangle(original_img_clone, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(original_img_clone, "OK", (x + 10, y + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
        print("\n\n****** Ink level is suffcient! ******")
    else:
        cv.rectangle(original_img_clone, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(original_img_clone, "NOT OK", (x + 10, y + 20), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
        print("\n\n****** Ink level is NOT sufficient! ******")
    
    show_image_center(cv, "The final result", original_img_clone)
        

import screeninfo

def show_image_center(cv, image_title, img):
    img_height = img.shape[0]
    img_width = img.shape[1]

    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height    
    x = int(screen_width / 2) - int(img_width /2)
    y = int(screen_height / 2) - int(img_height /2)
    
    # print("img_width: {}, img_height: {}".format(img_width, img_height))
    # print("screen_width: {}, screen_height: {}".format(screen_width, screen_height))
    # print("x: {}, y:{}".format(x, y))
    
    cv.namedWindow(image_title)
    cv.moveWindow(image_title, x, y)
    cv.imshow(image_title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


show_image = True

from picamera import PiCamera
def capture_image(cv, width, height, show_preview=True):    
    '''
    Captures an image by the given dimension and return an OpneCV image object
    
    '''
    print("# Step 1: Capturing an image of an ink barrel")
    camera = PiCamera(resolution=(1080, 1920), framerate=30)
    img = None
    try:
        if show_preview:
            camera.start_preview()
            sleep(2)
            camera.stop_preview()
        
        print("Start capturing an image")
        camera.capture('./images/pic.png', resize=(width,height))
        print("Finished capturing an image")
    finally:
        camera.close()
        print("A camera closed")


import RPi.GPIO as GPIO
button = 12
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(button, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
print("########### Start Ink level detector ######################")
print("Ink level detector initiated...")
print("Press the button to start detecting ink level")
try:
    while True:
        if GPIO.input(button) == GPIO.HIGH:
            print("The button pressed and start detecting ink level...")
                       
            capture_image(cv, 540, 960, show_image)
            original_img = read_captured_image(cv, show_image=show_image)
            img = convert_to_gray_image(cv, original_img, show_image)
            img = blur_image(cv, img, show_image)
            img = get_threshold(cv, plt, img, show_image)
            img = apply_opening(cv, img, show_image)
            contours = find_contours(cv, original_img, img, show_image)
            (contours, areas) = find_largest_contours(cv, original_img, contours, show_image)
            show_decision(cv, original_img, contours,0.17)

            print("\nInk level detection is done")
            sleep(1)
            print("press the button again if you want to detect another one")
finally:
    GPIO.cleanup()

print("################ Ink level detection finished #############################")


# Non-raspberry Pi version

'''
print("\n\n############## Start Ink level detector ver. non R-Pi ###############")

original_img = read_captured_image(cv, show_image=show_image)
img = convert_to_gray_image(cv, original_img, show_image)
img = blur_image(cv, img, show_image)
img = get_threshold(cv, plt, img, show_image)
img = apply_opening(cv, img, show_image)
contours = find_contours(cv, original_img, img, show_image)
(contours, areas) = find_largest_contours(cv, original_img, contours, show_image)
show_decision(cv, original_img, contours, 0.17)

print("\n\n############## Ink level detection finished #######################")
'''
