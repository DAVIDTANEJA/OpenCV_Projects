# detect 33 / 25 landmarks within a human body, with 24 fps
import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils   # use to draw landmarks
mpPose = mp.solutions.pose     # object
pose = mpPose.Pose()           # pose model

cap = cv2.VideoCapture(0)
pTime=0

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)  # send img to model
    # print(results.pose_landmarks)   # landmarks x,y,z values with 'visibility'

    # draw landmarks - results.pose_landmarks  ,  mpPose.POSE_CONNECTIONS - make/draw/ connections b/w landmarks
    if results.pose_landmarks:      # True
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # get these landmarks in a list, so we can find landmark positions 1,2....  and use acc.
        for id, lm in enumerate(results.pose_landmarks.landmark):
            # print(id, lm)      # give id/index with landmark(x,y,z)
            h,w,c = img.shape    # (height, width, channels) of image of particular landmark
            cx, cy = int(lm.x*w), int(lm.y*h)     # cx,cy - center position, values are in decimals so convert into int()
            # to check we are getting landmark acc. to id we create a big circle on id=0 landmark
            if id==0:
                cv2.circle(img, (cx,cy), 25, (255,0,255), -1)
            # Now we can put them in list and do sort of things


    # check frame rate fps -if video is fast and display it. can slow it down in waitKey(10/20) check, but when we use model automatically slow it down.
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)
 
    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27:    # 10 - for fps acc.
        break

# --------------------------------------------------------------------------------------------------------------
# Now convert this into module ,  create file :  "pose_estimaion_mosule.py"


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, detectionCon=0.5, trackCon=0.5):    # these parameters are of pose()
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # initialize object
        self.mpDraw = mp.solutions.drawing_utils   # use to draw landmarks
        self.mpPose = mp.solutions.pose     # object
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)  # pose model

    # find pose method - draw landmarks
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)  # send img to model
        # print(results.pose_landmarks)   # landmarks x,y,z values with 'visibility'

        # draw landmarks - results.pose_landmarks  ,  mpPose.POSE_CONNECTIONS - make/draw/ connections b/w landmarks
        if self.results.pose_landmarks:      # True
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    # get these landmarks in a list, so we can find landmark positions 1,2....  and use acc.
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)      # give id/index with landmark(x,y,z)
                h,w,c = img.shape    # (height, width, channels) of image of particular landmark
                cx, cy = int(lm.x*w), int(lm.y*h)     # cx,cy - center position, values are in decimals so convert into int()
                lmList.append([id, cx, cy])   # append in a list

                # to check we are getting landmark acc. to id we create a big circle on id=0 landmark
                if id==0:
                    cv2.circle(img, (cx,cy), 25, (255,0,255), -1)
        return lmList


def main():
    cap = cv2.VideoCapture(0)
    pTime=0
    detector = poseDetector()    # here we create obejct for class function
    while True:
        _, img = cap.read()
        # inside while loop we call methods
        img= detector.findPose(img)           # draw=False
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            # print(lmList)    # lmList[0]  - for particular point
            # we can also track particular point and draw circle
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 25, (255,0,255), -1)

        # check frame rate fps -if video is fast and display it. can slow it down in waitKey(10/20) check, but when we use model automatically slow it down.
        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)
 
        cv2.imshow("image", img)
        cv2.waitKey(1)    # 10 - for fps acc.


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------------------------------------------------------------
# After creating module we can use this into any projects
# create file :  "pose_projects.py"

import PoseEstimationModule as pm    # module

cap = cv2.VideoCapture(0)
pTime=0
detector = pm.poseDetector()    # call the module
while True:
    _, img = cap.read()
    # inside while loop we call methods
    img= detector.findPose(img)           # draw=False
    lmList = detector.findPosition(img)
    # if len(lmList) != 0:
    #     # print(lmList)    # lmList[0]  - for particular point
    #     # we can also track particular point and draw circle
    #     cv2.circle(img, (lmList[14][1], lmList[14][2]), 25, (255,0,255), -1)

    # check frame rate fps -if video is fast and display it. can slow it down in waitKey(10/20) check, but when we use model automatically slow it down.
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,0), 3)
 
    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27:   # 10,20 - use for fps acc.
        break

