# Here we detect form and cut the form image out and detect 'ROI' - text filled into the form and put into csv file as output.
# we need :  1 'query' image : which is blank form  ,  2.test images -3or 4 : "filled forms"  - where we get the data

# 1st to get 'ROI' where details filled, we have code below 1st run that code and get all ROI points and provide here , 
# Remeber it asks : 1.for "type" : enter "text/box" as required , 2. type "Name" : enter "Name/Phone/Email/City" for which required , After getting all ROI double/single click on top of image and press "s" it will give lists of all ROI, copy and paste it here.

import cv2
import numpy as np
from numpy.core.defchararray import count
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' 
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'


# To get ROI , we have code below 1st run get all points, Remeber it asks : 1.for "type" : enter "text/box" as reuired , 2. type "Name" : enter "Name/Phone/Email/City" as required , After getting all ROI double/single click on top of image and press "s" it will give lists of all ROI, copy and paste it here.
# [x,y,w,h] , text input / check box , Name/ Phone no.
roi = [[]]    # paste here , you can change here text spelling here if any.


imgQ = cv2.imread("Query.png")
h,w,c = imgQ.shape

# detector - ORB , bcoz its free
orb = cv2.ORB_create(1000)    # 1000 - features/key points -which it will detect on image , change acc.
kp1, des1 = orb.detectAndCompute(imgQ, None)   # kp key points - unique points of image , descriptor - representation of key points which are easier for computer to understand and differntiate , None - is mask
# imgkp1 = cv2.drawKeypoints(imgQ, kp1, None)   # None - here comes output image but now we are storing in var. - 'imgkp1'
# cv2.imshow('key points', imgkp1)

# Now bring test images so we can find featurs of these images and then compare it with features of query image - by matching
per = 25     # percentage
pixelThreshold = 500   # pixels value - checking if value above checkbox filled , if below checkbox not filled.
path = "UserForms"      # testing image folder
myPicList = os.listdir(path)
for j,y in enumerate(myPicList):        # j- form no., y- form image
    img = cv2.imread(path + "/" + y)    # complete path for image
    # cv2.imshow(y, img)
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)       # bf - brute force , to match images
    matches = bf.match(des2, des1)             #  here we match descriptor - images
    matches.sort(key = lambda x : x.distance)  # sort all matches based on distance, lower distane - better match , sorting : we will have 1st good matches , 2nd bad matches
    good = matches[:int(len(matches)*(per/100))]   # good matches , increase acc.
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:100], None, flags=2)   # draw matches points b/w img, imgQ , good[:100] - good matches -change acc.
    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1,1,2)   # getting all points from good list and then actual points from them
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)  # find relationship, M- Matrix
    imgScan = cv2.warpPerspective(img, M, (w,h))    # use this matrix to align form - cutout form image only, also form is distorted from somewhere it will fill black there.
    # cv2.imshow(y, imgScan)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)    # create mask for image
    # loop though all ROI , create rectangle on this mask image using ROI and overlay it on original image
    myData = []   # ROI filled data of images
    for x,r in enumerate(roi):    # r[0,1,2,3] = [(x,y),(w,h), text/checkbox, Name/Phone no./City]
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0,255,0), -1 )   # (x,y) , (w,h)   # select ROI region and crop it and send it to pytesseract
        imgShow = cv2.cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]   # [h, w] , crop out the required ROI image part
        # cv2.imshow(str(x), imgCrop)

        if r[2] == "text":   # check ROI is text / check box
            myData.append(pytesseract.image_to_string(imgCrop))
            # print(f"{r[3]} : {pytesseract.image_to_string(imgCrop)}")

        # here we checking checkbox if filled - it will give some pixels values  ,  if not filled - gives 0
        if r[2] == "box":
            imgGray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGray, 170, 255, cv2.THRESH_BINARY_INV)[1]   # [1] - 2nd element, Inversing - dark region gives 1 and bright region gives 0
            totalPixels = cv2.countNonZero(imgThresh)    # how many pixels are dark converted to 1
            # print(totalPixels)
            if totalPixels > pixelThreshold:
                totalPixels = 1       # if checkbox filled
            else:
                totalPixels = 0       # if not filled 

            print(f"{ r[3] } : {totalPixels} ")
            myData.append(totalPixels)

        # Display text detected on original image - name, phone, city etc.
        cv2.putText(imgShow, str(myData[x]), (r[0][0], r[0][1]), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 2)

    # Now we save data in csv file "Data_Output.csv" , Remember : to write heading names manually
    with open("files/Data_Output.csv", "a+") as f:
        for data in myData:
            f.write(str(data) + ",")
        f.write('\n')    # save in new line


    cv2.imshow(y +"2", imgShow)


cv2.imshow("output", imgQ)
cv2.waitKey(0)



# ----------------------------------------------------------------------------
# ROI : 
import cv2, random

path = 'Query.png'
scale = 0.5
circles = []
counter = 0
counter2 = 0
point1 = []
point2 = []
myPoints = []
myColor = []

def mousePoints(event, x,y, flags, params):
    global counter, point1, point2, counter2, circles, myColor
    if event == cv2.EVENT_LBUTTONDOWN:
        if counter == 0:
            point1 = int(x//scale), int(y//scale)
            counter += 1
            myColor = (random.randint(0,2)*200, random.randint(0,2)*200, random.randint(0,2)*200)
        elif counter == 1:
            point2 = int(x//scale), int(y//scale)
            type = input("Enter type")
            name = input("Enter Name")
            myPoints.append([point1, point2, type, name])
            counter = 0
        circles.append([x,y, myColor])
        counter2 += 1

img = cv2.imread(path)
img = cv2.resize(img, (0,0), None, scale, scale)

while True:
    for x,y,color in circles:
        cv2.circle(img, (x,y), 3, color, -1)
    cv2.imshow("Original image", img)
    cv2.setMouseCallback("Original image", mousePoints)    # both name should be same "Original image"

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(myPoints)
        break



