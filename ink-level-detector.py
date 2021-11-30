import cv2 as cv
from matplotlib import pyplot as plt
import imutils

# Step 1: resize image
def standardize_image(cv, imagePath, show_image):
    '''
    Convert a file to graysacle and reszie it to 400 x height

    '''
    # Step 1: convert image jgp to png
    img = cv.imread("./test.jpg")
    cv.imwrite("./test.png", img)
    png_img = cv.imread("./test.png")

    # Step 2: resize an image
    width = 400
    height = int(width * png_img.shape[0] / img.shape[1])
    print("width: " + str(width))
    print("height: " + str(height))
    resized_img = cv.resize(png_img, (width, height), interpolation = cv.INTER_AREA)
    
    if show_image:
        cv.imshow("Resized PNG Image", resized_img)
        cv.waitKey(0)
    
    
    # Step 2: chnage it to grayscale and resize image
    # img = cv.imread("./test.png")
    
    # width = 400
    # height = int(width * int(img.shape[0]) / int(img.shape[1]))

    # print("width: " + str(width))
    # print("height: " + str(height))
    
    # resized_img = cv.resize(img, (width, height), interpolation = cv.INTER_AREA)
    # gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    # cv.imwrite("./test_resized.png", gray_img)
    
    return resized_img

def blur_image(cv, img):
    img = cv.GaussianBlur()
# Step 1: convert an image to an grayscale image
# def convert_to_grayscale(cv, imagePath):
#     '''
#     convert a regular image to gray scale image

#     cv -- the referece of cv2 object
#     imagePath -- the image path string 
#     '''

print("convert images")
standardize_image(cv, "hi", True)


# pen_image = cv.imread("./images/pen1.png")
# pen_gray = cv.split(pen_image)[0]
# cv.imshow("Gray Image", pen_gray)
# cv.waitKey(0)


