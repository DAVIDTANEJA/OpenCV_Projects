# Virtual paint using "Hand Tracking Module"
# goto : canva.com - helps to design brochures / business cards etc. : signup 
# we take 4 same pictures , but there 1 difference in every image -just to handle events.

# Steps : 1.import images , 2.find hand landmarks , 3. detect/ check which finger is up -like we count fingers 
# 4. select mode : 2 fingers to selection of color/eraser) 5. drawing mode : 1 finger for color paint , so we can easily move around canvas


import cv2
import numpy as np
import mediapipe as mp
import os, time
import HandTrackingModule as htm

# read the images, Remeber we have 4 images and we change acc. to selection of each image with particular color paint functionality.
folderpath = "Header"
myList = os.listdir(folderpath)   # get the folder in list
# print(myList)
overlayList = []
for imPath in myList:                               # read image from list
    image = cv2.imread(f"{folderpath}/{imPath}")    # create path for the images
    overlayList.append(image)                       # and append into list so we can use it.
# print(len(overlayList))

header = overlayList[0]    # overlay image on webcam, by default header is 1ts image
drawColor = (255,0,255)    # whenever the value of color change we change it to this color

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.7)   # change 'confidence' for more accuracy

brushThickness = 15
eraserThickness = 50
xp, yp = 0, 0
imgCanvas = np.zeros((720,1280,3), np.uint8)   # when we drawing image update and it deletes what we draw, so using this canvas image so iprevious part will not deleted.

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)    # flip : bcoz when we draw on left side it appears on right side, so we go right side it goes right side and vice-versa.
    
    # 2. find hand landmarks of - index finger and middle finger
    img= detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)
        x1, y1 = lmList[8][1:]     # list contains [id, x, y] and  we need (x,y)  of : tip of index finger - 8 and middle fingers
        x2, y2 = lmList[12][1:]    # middle finger

        # 3. detect which finger is up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4.Selection mode: 2 fingers to select color/eraser
        if fingers[1] and fingers[2]:       # if 2 fingers up 1st and 2nd
            xp, yp = x1, y1            # whenever the hand is detected it will start drawing from there

            # cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, -1)   # rectnagle in selection when we draw we use circle
            # if we are on top means in selection mode of : color/eraser  ,  1280 x 720
            if y1 < 125:      # header image height
                if 250 < x1 < 450:             # then we divide '1280' for 'x' for particular color / eraser
                    header = overlayList[0]    # and change image acc. to selection of particular color , like clicking 1st color on image
                    drawColor = (127,0,255)    # pink color
                elif 550 < x1 < 750:           # for blue color   ,  Remember : check the range then range for 'x'
                    header = overlayList[1]    
                    drawColor = (0,0,255)      
                elif 800 < x1 < 950:           # for green color
                    header = overlayList[2]    
                    drawColor = (127,0,255)    
                elif 1050 < x1 < 1200:         # eraser - black color
                    header = overlayList[3]    
                    drawColor = (0,0,0)
              # print("Selection mode")
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, -1)    # it will selected color otherwise purple

        # 5.Drawing mode : we ue 2 methods : circle(here it lacks some space) and line (here we need starting and ending point)
        if fingers[1] and fingers[2] == False:   # means both fingers are not up
            cv2.circle(img, (x1,y1), 15, drawColor, -1)      # draw circle on finger tip

            if xp == 0 and yp == 0:
                xp, yp = x1, y1        # it will take previous point / starting point where we want to start from , not from 0,0

            if drawColor == (0,0,0):    # if we are using eraser increase thickness of eraser
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)   # xp - previous position of 'x'
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)   # using 'imgCanvas' bcoz when we draw on 'img' it updates and deletes previous drawing
            
            xp, yp = x1, y1    # updates the x,y positions

            # print("Drawing mode")

    # drawing on original image instead of canvas image
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)   # 1st convert into gray image
    # what it will do , masking : convert black region into white and color part into black
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)    # then convert into binary image and then inverse it
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)    # converting back to merge with img
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # set image on webcam
    img[0:125, 0:1280] = header   # img[height , width]  overlay same as image size , our image size = 1280x125
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)   # we can add both of them, so drawing can occur on original image

    cv2.imshow("image", img)
    cv2.imshow("Canvas image", imgCanvas)
    cv2.imshow("Inverse image", imgInv)
    if cv2.waitKey(1) == 27:
        break




