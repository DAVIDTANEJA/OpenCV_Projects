from typing import final
from PIL.Image import ImageTransformHandler
import cv2
import numpy as np

def detectColor(img, hsv):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow("HSV", imgHSV)
    lower = np.array([hsv[0], hsv[2], hsv[4]])
    upper = np.array([hsv[1], hsv[3], hsv[5]])
    mask = cv2.inRange(imgHSV, lower, upper)          # text detected in black-white color
    imgResult = cv2.bitwise_and(img, img, mask=mask)  # original image with highlighted text detected
    return imgResult


def getContours(img, imgDraw, cThr=[100,100], showCanny=False, minArea=1000, filter=0, draw=False):
    imgDraw = imgDraw.copy()    # copy it not to affect original image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.array((10,10))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)    # increase thickness
    imgClose = cv2.morphologyEx(imgDial, cv2.MORPH_CLOSE, kernel)  # close gaps b/w contours

    # if showCanny True
    if showCanny: cv2.imshow('Canny', imgClose)
    contours, hierarchy = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # give all contours
    finalContours = []
    for i in contours:              # and loop thorigh all contours and check area
        area = cv2.contourArea(i)
        if area > minArea:                 # if its exceed 'minArea', used to remove noise
            peri = cv2.arcLength(i, True)  # find perimeter
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)    # how many corners points you have, if have '4' its rectangle/squre , '3' its rectangle
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:      #check len of 'approx' what we need and  'filter' we define to find shape like rectangle, so we use "filter=4" where we use this function
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                finalContours.append([len(approx), area, approx, bbox, i])    # if we don't define any 'filter' value then it will send output of all contours

    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)    # sort all contours based on size

    if draw:
        for con in finalContours:
            x,y,w,h = con[3]          # con[3] in 'finalContours' is - bbox  - to draw
            cv2.rectangle(imgDraw, (x,y), (x+w, y+h), (255,0,255), 2)
            cv2.drawContours(imgDraw, con[4], -1, (0,0,255), 2)
    
    return imgDraw, finalContours       # these 'finalContours' are we use to crop image


# get roi from image using contours and store in a list
def getRoi(img, contours):
    roiList = []
    for con in contours:
        x,y,w,h = con[3]        # con[3] is bbox
        roiList.append(img[y:y+h, x:x+w])   # crop 'roi' part from 'img'
    return roiList

# display roi images from roiList contours
def roiDisplay(roiList):
    for x, roi in enumerate(roiList):
        roi = cv2.resize(roi, (0,0), None, 2, 2)    # resize roi images bcoz its small
        cv2.imshow(str(x), roi)      # 'x' how many loops it done to name our images of roi 0,1,2....


# save the text from roi images
def saveText(highLightedText):
    with open("HighLightedText.csv", 'w') as f:
        for text in highLightedText:
            f.writelines(f"\n{text}")


def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver
