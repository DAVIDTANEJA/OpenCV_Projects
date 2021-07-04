# module created from "Hand_Tracking_Detection.py"  so we can use it in different functions like : Gesture volume control
import cv2
import time
import mediapipe as mp

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

        self.tipIds = [4,8,12,16,20]    # tips of fingers : thumb,index,so on..


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
        # draw bound box around hand
        xList = []
        yList = []
        bbox = []
        # finding position of landmarks, we can put them in list and return this list and do sort of things.
        self.lmList = []
        # Find 1st hand, then find all landmarks and put them in list
        if self.results.multi_hand_landmarks:            # True , check for getting the landmarks
            myHand = self.results.multi_hand_landmarks[handNo]    # find 1 hand
            # find landmarks , getting information using id/index, landmark has (x-y-z cordinates) , id exact relates to landmark no., we use x-y cordinates to find the location for the landmark on hand. but x-y values in decimals so location in pixels. so we multiply with width,height get the pixel value.
            for id, ln in enumerate(myHand.landmark):
                # print(id, ln)
                h,w,c = img.shape    # (height, width, channels) of image
                cx, cy = int(ln.x*w), int(ln.y*h)     # cx,cy - center position, values are in decimals so convert into int()
                xList.append(cx)    # xlist, ylist - to draw bbox around hand
                yList.append(cy)
                # print(id, cx, cy)   # tells center point with id
                self.lmList.append([id, cx, cy])   # put them in a list
                # to check we are getting landmark acc. to id we create a big circle on id=0 landmark
                # if draw:
                    # cv2.circle(img, (cx,cy), 25, (255,0,255), -1)

            # draw bbox
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0,255,0), 2)

        return self.lmList, bbox

    # count fingers up
    def fingersUp(self):
        fingers = []   # save the finger open/close
        # To count fingers, we take tip landmark of fingers like 8 and to count apply condition if its below then point 7/6/5 then finger is closed, so don't count it. opencv orientation tells - finger up means lower value, max height 0
        # 1-5 for index to small finger , not thumb which is '0' bcoz it does not goes below -2 'thumb' moves left side when we down it, we use for loop for finger tip Ids, if there are many scenarios like - 'thumb' can apply if-conditons        
        # Remember this for Right hand thumbm, it reverse this condition for left hand thumb loop
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0]-1][1]:   # for thumb Id '0' , list contains [id, x, y], so here taking for x [1] - to move thumb left side., tipIds[id] -1 : 1 landmark point below
            fingers.append(1)
        else:
            fingers.append(0)
        # fingers
        for id in range(1,5):    # 1-5 for index to small  finger , here we use list - 'y'axis
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:    # tipIds[id] : 8-index finger tip, list contains [id, x, y], so here taking for y [2]  , and tipIds[id]-2 means : we taking point 6 (or below 2 landmark point) for closing the finger, and it will loop through all ids, we need to save also finger opens/close in list
                fingers.append(1)     # if finger open append - 1
                # print("index finger open")
            else:
                fingers.append(0)        # if finger close append - 0
        # print(fingers)

        return fingers



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
