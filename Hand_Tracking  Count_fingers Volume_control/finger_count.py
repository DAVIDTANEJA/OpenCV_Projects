# count fingers using module "Hand Tracking module" created.
# Displaying numbers acc. to fingers , also change images of fingers acc. to detection of fingers

import cv2
import time, math, os
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

# folder containing images of fingers - display on webcam on left side, images size is 200x200
# folderpath = "fingerImages"
# myList = os.listdir(folderpath)
# overlayList = []
# for imgPath in myList:
#     image = cv2.imread(f"{folderpath}/{imPath}")   # get path of image 1 by 1
#     overlayList.append(image)

detector = htm.handDetector(detectionCon=0.7)
tipIds = [4,8,12,16,20]    # tips of fingers : thumb,index,so on..
pTime = 0

while True:
    _, img = cap.read()
    img = detector.findHands(img)    # detect hand
    lmList = detector.findPosition(img, draw=False)   # list of landmarks, draw=Fals bcoz we alrady drawing in findHands()
    if len(lmList) != 0:
        fingers = []   # save the finger open/close
        # To count fingers, we take tip landmark of fingers like 8 and to count apply condition if its below then point 7/6/5 then finger is closed, so don't count it. opencv orientation tells - finger up means lower value, max height 0
        # 1-5 for index to small finger , not thumb which is '0' bcoz it does not goes below -2 'thumb' moves left side when we down it, we use for loop for finger tip Ids, if there are many scenarios like - 'thumb' can apply if-conditons        
        # Remember this for Right hand thumbm, it reverse this condition for left hand thumb loop
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:   # for thumb Id '0' , list contains [id, x, y], so here taking for x [1] - to move thumb left side., tipIds[id] -1 : 1 landmark point below
            fingers.append(1)
        else:
            fingers.append(0)            
        # for fingers
        for id in range(1,5):    # 1-5 for index to small  finger , here we use list - 'y'axis
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:    # tipIds[id] : 8-index finger tip, list contains [id, x, y], so here taking for y [2]  , and tipIds[id]-2 means : we taking point 6 (or below 2 landmark point) for closing the finger, and it will loop through all ids, we need to save also finger opens/close in list
                fingers.append(1)     # if finger open append - 1
                # print("index finger open")
            else:
                fingers.append(0)        # if finger close append - 0
        # print(fingers)    # print list of 0 and 1 acc. to fingers open-close

        # Change images acc. to fingers count
        totalFingers = fingers.count(1)   # counts 1 in list how many 1 are there and change img. acc.
        print(totalFingers) 
        # h,w,c = overlayList[totalFingers-1].shape        # totalFingers-1 : means it will take 0th image from 'overlayList' list and -1 means 0th element i.e. we take 6th image for this closed fingers image.
        # img[:200, :200] = overlayList[totalFingers-1]    # display image on webcam left side, image size 200x200

        # display numbers acc. to images / totalFingers count
        cv2.rectangle(img, (20,225), (170, 425), (0,255,0), -1)
        cv2.putText(img, str(totalFingers), (45,375), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 25)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (400,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27:
        break





