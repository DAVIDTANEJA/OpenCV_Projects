# Bicep curls : we take 3 points (wrist, elbow, shoulder) and fild angle for bicep curl and do calculation for posture and count on that also tell correct/not.
# we define a function for many exercise which we find based on angle to calculate many postures.

import cv2
import mediapipe as mp
import time
import PoseEstimationModule as pm
import numpy as np


cap = cv2.VideoCapture(0)    # try video clip , for webcam maintain the distance otherwise it will not capture properly

detector = pm.poseDetector()
# now we count the dumbbell curls/wraps of exercise do
count = 0
dir = 0      # direction : 0 -when it goes up , 1 for down , consider full 1 curl done when both of these 0,1, so means 0 to 100 and 100 to 0 acc. to percentage range declared.
pTime = 0    # previous time for 'fps'

while True:
    _, img = cap.read()

    # img = cv2.imread('tricep.jpg')        # try with image
    img = detector.findPose(img, False)           # False - not draw the pose, in below "findAngle()" when put 3 points we get pose of that only
    lmList = detector.findPosition(img, False)    # find landmarks 
    # print(lmList)
    if len(lmList) != 0:
        angle = detector.findAngle(img, 12,14,16)    # tricep (left hand-11,13,15) (right hand-12,14,16)  , left leg- 23,25,27 , right leg- 24,26,28
        # angle = detector.findAngle(img, 11,13,15)    # left hand

        # Now we convert curl/angle into percentage, Remeber this angle range which we are converting to 0 to 100 -decide count of curls
        per = np.interp(angle, (210,310), (0,100))    # (210,310) - angle range making change acc. , (0,100) converting angle range into %
        # print(angle, per)
        # check for 1 complete dumbbell curl means  0 to 100 , 100 to 0, => or 0 means going 'up' and reaches 100% , Then we change the direction downwards and vice-versa , 1 means going 'down' reaches 100 % acc. there then 1 curl completed.
        if per == 100:
            if dir == 0:        # 0 means going up
                count += 0.5
                dir == 1         # change the direction
        if per == 0:
            if dir == 1:
                count += 0.5
                dir == 0
        # print(count)

        # display count of curls
        cv2.rectangle(img, (50,200), (200,350), (0,255,0), -1)
        cv2.putText(img, str(int(count)), (70,300), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 5)

    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27:
        break
