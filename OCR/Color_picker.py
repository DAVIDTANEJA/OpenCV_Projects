# grab particular color part of image, we don't know the min-max. color range to detect,
# so we use 'trackbars' which help to detect in real time color range to grab the part of image.
import cv2
import numpy as np

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



# Read Image
path = "abc.png"       # take image acc.

# HSV - hue saturation value limits - define color values/ranges in which we want our color to be. If the 'image region' falls into that 'color range' we will grab that image part.
# basically grab particular color part of image, we don't know the min-max. color range to detect, so we use 'trackbars' which help to detect in real time color range to grab image part.
def empty(a):
    pass

# create new window , both name should be same of "window and resize"
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640,240)
# create trackbar: we need 6 values : Hue min., hue max., saturation min., saturation max., value min., value max.  ,  # hue max. value is 360, but in opencv we have max.-179
# (min., max. value, function which run every time if user/something change in trackbar)
# 1st run these trackbars, after getting mask values, comment these trackbars out / change values
# cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
# cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
# cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
# cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
# cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

# 2nd run these TrackBars , after when running program getting mask value , change values min .acc.
# Now if we run these values changed, we will get image of selelcted color part 
cv2.createTrackbar("Hue Min", "TrackBars", 29, 179, empty)  # 29
cv2.createTrackbar("Hue Max", "TrackBars", 49, 179, empty) # 49
cv2.createTrackbar("Sat Min", "TrackBars", 209, 255, empty) # 209
cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty) # 255
cv2.createTrackbar("Val Min", "TrackBars", 56, 255, empty) # 56
cv2.createTrackbar("Val Max", "TrackBars", 103, 255, empty) # 103

# Now we read trackbar values, so we can apply on image
# in order to get value we need to put in a loop, to keep getting that value, instead of image we have to change to webcam
while True:
    img = cv2.imread(path)                         # take any image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")  # spelling should be same (min, to which trackbar window does it belong)
    h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min = cv2.getTrackbarPos("Val Min","TrackBars")
    v_max = cv2.getTrackbarPos("Val Max","TrackBars")
    print(h_min, h_max)
    # Now we use these values to filter out particular color part of image
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv_img, lower, upper)   # filter out image color
    # if run program, we get mask window and if we change trackbar value we get image, So we keep all colors we don't want as black, what we need color make it white and note down the values from trackbar.
    # Now go above where we created Trackbar and change values for : "Hue min, max, Sat min, max, Val min, max" , and After changing values run program we get selected part image black-white mask
    # To get original image from 'mask image', we will create new image 
    result_img = cv2.bitwise_and(img,img, mask=mask)

    # In the end we use "stack images()" function where we change "Trackbar values" real time and get the "mask image" values and 'result image'
    stack_img = stackImages(0.6, ([img,hsv_img], [mask, result_img]))  # scale, array of images 1st row, 2nd row
    cv2.imshow('Stack Images', stack_img)           # display image

    # cv2.imshow('Original', img)           # display image
    # cv2.imshow('HSV', hsv_img)
    # cv2.imshow('mask', mask)
    # cv2.imshow('Result', result_img)

    cv2.waitKey(1)

# Note : when run program we have :   1.img - original image shown , hsv_img, 
# 3.mask : in which we convert white-black color acc. to needed color ,  
# 4.result_img : in which we see original color needed. 
