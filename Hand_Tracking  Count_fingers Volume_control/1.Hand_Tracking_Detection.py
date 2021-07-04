# pip install mediapipe  - 'mediapipe' model created by Google
# 1st detect and make connections : Each hand - have 21 points to detect
# 2nd track positions and perform task by creating list - by getting information using id, landmark(x-y cordinates)
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands   # create object
hands = mpHands.Hands()        # detect and track hands
mpDraw = mp.solutions.drawing_utils  # draw lines b/w 21 points of each hand

pTime = 0  # previous time
cTime = 0  # current time

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # hands - uses RGB images
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)  # check - hands tracking gives value otherwise None

    if results.multi_hand_landmarks:   # True
        for handLns in results.multi_hand_landmarks:    # handLns - 1 hand get information
            # getting information using id/index, landmark has (x-y-z cordinates) , id exact relates to landmark no., we use x-y cordinates to find the location for the landmark on hand. but x-y values in decimals so location in pixels. so we multiply with width,height get the pixel value.
            for id, ln in enumerate(handLns.landmark):
                print(id, ln)      # give id/index with landmark(x,y,z)
                h,w,c = img.shape    # (height, width, channels) of image
                cx, cy = int(ln.x*w), int(ln.y*h)     # cx,cy - center position, values are in decimals so convert into int()
                # print(id, cx, cy)   # tells center point with id
                # to check we are getting landmark acc. to id we create a big circle on id=0 landmark
                # if id==0:
                #     cv2.circle(img, (cx,cy), 25, (255,0,255), -1)
                # Now we can put them in list and do sort of things

            mpDraw.draw_landmarks(img, handLns, mpHands.HAND_CONNECTIONS)  # handLns -draw landmarks on BGR image - 1 hand, mpHands.HAND_CONNECTIONS -make connections b/w them
    # check frame rate fps -if video is fast
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime   # then previous time will become current time
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)  # put on screen - (img, int(fps), x,y-position, font, scale, color, thickness)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()


# --------------------------------------------------------------------------------------
# Now we create module from this above tracking/detecting hand landmarks.
# So if we use this in any project, don't need to write all of it again.
# we can take 'list of the 21 values' of each hand  and perform.
# create file :  "hand_tracking_module.py"  and use this code:
# we can also add more methods acc. to needs ,  like :  findHands , findPosition.

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5 ):    # hand detector parameters, which are in mediapipe Hands() by default, here we can manipulate
        self.mode = mode    # create obejct for var -'self.mode'  and assigning value provided by user 'mode'
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # initialize  and  we now use every var. as - "self.mode" using self
        self.mpHands = mp.solutions.hands   # create object
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)        # detect and track hands
        self.mpDraw = mp.solutions.drawing_utils  # draw lines b/w 21 points of each hand

    # detection  and  we now use every var. as - "self.mode" using self , # draw=True put landmarks on hands, to sue in all methods we use self.mode / self.results
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # hands - uses RGB images
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)  # check - hands tracking gives value otherwise None

        if self.results.multi_hand_landmarks:   # True
            for handLns in self.results.multi_hand_landmarks:    # handLns - give 1 hand information, but we are getting here for all hands if we shows
                if draw:          # True
                    self.mpDraw.draw_landmarks(img, handLns, self.mpHands.HAND_CONNECTIONS)  # handLns -draw landmarks on BGR image - 1 hand, mpHands.HAND_CONNECTIONS -make connections b/w them
        return img

    # find positions of landmarks, handNo - whichever hand information 1,2 we want
    def findPosition(self, img, handNo=0, draw=True):
        # finding position of landmarks, we can put them in list and return this list and do sort of things.
        lmList = []
        # Find 1st hand, then find all landmarks and put them in list
        if self.results.multi_hand_landmarks:            # True , check for getting the landmarks
            myHand = self.results.multi_hand_landmarks[handNo]    # find 1 hand
            # find landmarks , getting information using id/index, landmark has (x-y-z cordinates) , id exact relates to landmark no., we use x-y cordinates to find the location for the landmark on hand. but x-y values in decimals so location in pixels. so we multiply with width,height get the pixel value.
            for id, ln in enumerate(myHand.landmark):
                # print(id, ln)
                h,w,c = img.shape    # (height, width, channels) of image
                cx, cy = int(ln.x*w), int(ln.y*h)     # cx,cy - center position, values are in decimals so convert into int()
                # print(id, cx, cy)   # tells center point with id
                lmList.append([id, cx, cy])   # put them in a list
                # to check we are getting landmark acc. to id we create a big circle on id=0 landmark
                # if draw:
                    # cv2.circle(img, (cx,cy), 25, (255,0,255), -1)

        return lmList


# whatever we write in 'main' will be dummy code , used to show what can this module do.
def main():
    pTime = 0  # previous time
    cTime = 0  # current time
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        success, img = cap.read()   # once we get the image then we call method of handDetector to find hands

        img = detector.findHands(img)       # calling method findHands() - track hand
        lmList = detector.findPosition(img)  # calling method findPosition()  - gives list of landmarks
        if len(lmList) != 0:    # if list not 0, that means we showing hand 
            print(lmList[0])    # checking for 0 index/id, find any

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime   # then previous time will become current time
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)  # put on screen - (img, int(fps), x,y-position, font, scale, color, thickness)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------------------------------------------------------------------------
# After creating module we can use this module in different projects.
# create file :  "hand_tracking_game.py"  and  "import module" here

import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0  # previous time
cTime = 0  # current time
cap = cv2.VideoCapture(0)
detector = htm.handDetector()    # now use like this : htm.handDetector()

while True:
    success, img = cap.read()   # once we get the image then we call method of handDetector to find hands

    img = detector.findHands(img, draw=False)       # calling method findHands() - track hand, here 'False' means it will not tarck/detect and draw on hand landmarks
    lmList = detector.findPosition(img, draw=False)  # calling method findPosition() - gives list of landmarks, here False will not draw circle we created in method.
    if len(lmList) != 0:    # if list not 0, that means we showing hand position x,y through id
        print(lmList[0])    # checking for 0 index/id, find any

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime   # then previous time will become current time
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_COMPLEX, 3, (255,0,255), 3)  # put on screen - (img, int(fps), x,y-position, font, scale, color, thickness)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break

