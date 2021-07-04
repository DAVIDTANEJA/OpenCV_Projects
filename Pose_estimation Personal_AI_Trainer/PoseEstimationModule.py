import cv2
import mediapipe as mp
import time, math

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
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                # print(id, lm)      # give id/index with landmark(x,y,z)
                h,w,c = img.shape    # (height, width, channels) of image of particular landmark
                cx, cy = int(lm.x*w), int(lm.y*h)     # cx,cy - center position, values are in decimals so convert into int()
                self.lmList.append([id, cx, cy])   # append in a list

                # to check we are getting landmark acc. to id we create a big circle on id=0 landmark
                # if id==0:
                #     cv2.circle(img, (cx,cy), 25, (255,0,255), -1)
        return self.lmList

    # find angle b/w landmarks (we need 3 points: p1,p2,p3) - these are basically id/index no. of list of landmarks. so p1 has list of [id, x, y] we take x,y acc. to our needed 'id'
    def findAngle(self, img, p1, p2, p3, draw=True):
        # get the landmarks
        x1, y1 = self.lmList[p1][1:]     # p1 has [id,x,y] and this [1:] - will give x,y
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # calculate angle
        angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
        # print(angle)
        # if angle goes negative '-ve' then we add 360
        if angle < 0:
            angle += 360

        # draw line and on line draw all circle - 3 points / landmarks we needed  ,  draw angle on image
        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,255,255), 3)  # line b/w p1-p2 point
            cv2.line(img, (x2,y2), (x3,y3), (255,255,255), 3)  # line b/w p2-p3 point
            cv2.circle(img, (x1,y1), 10, (0,0,255), -1)        # circle on p1
            cv2.circle(img, (x1,y1), 15, (0,0,255), 2)         # create outer circle on p1
            cv2.circle(img, (x2,y2), 10, (0,0,255), -1)        # p2 
            cv2.circle(img, (x2,y2), 15, (0,0,255), 2)
            cv2.circle(img, (x3,y3), 10, (0,0,255), -1)        # p3
            cv2.circle(img, (x3,y3), 15, (0,0,255), 2)

            # cv2.putText(img, str(int(angle)), (x2+40, y2-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)  # put angle , play with this position : (x2+40, y2-10)

        return angle


def main():
    cap = cv2.VideoCapture(0)
    pTime=0
    detector = poseDetector()    # here we create obejct for class function
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
        if cv2.waitKey(1) == 27:    # 10 - for fps acc.
            break

if __name__ == "__main__":
    main()
