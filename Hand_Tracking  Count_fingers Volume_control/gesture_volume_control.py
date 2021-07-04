# control the volume of computer using - "Hand Tracking Module"  and  also using 'pycaw' library to control system volume.

import cv2
import time, math
import numpy as np
import mediapipe as mp
import HandTrackingModule as htm     # 'module' should be in same folder
# Volume control imports             # pip install pycaw
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


# parameters
wCam, hCam = 640, 480      # 1280,720
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# 1st create object, 2nd we here change 'detectionCon' to be really sure about hand then it detect hand. so it will work smoothly
detector = htm.handDetector(detectionCon=0.7)

# To control the volume - use this code
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()                 # which no need -comment out
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()                # 1st print range : print(volume.GetVolumeRange()) : gives (-65.25, 0.0, 0.03125)  -> min :-65, max : 0
minVol = volRange[0]    # then we set min max vol.
maxVol = volRange[1]


pTime = 0
vol = 0
volBar = 400      # acc. to volBar : vol= 0 at 400 , Remember this 'volBar' value act as default value for vol.
volPer = 0
while True:
    _, img = cap.read()

    # methods
    img = detector.findHands(img)   # 1st find / detect hand
    lmList = detector.findPosition(img, draw=False)   # 2nd get position of landmarks, False -bcoz we are already drawing it.
    if len(lmList) != 0:        # so the list not empty otherwise it will throw error 
        # print(lmList)       # 21 points list , we can get particular point - print(lmList[4], lmList[8]) : we use here 4-thumb tip, 8-index finger tip - to control volume
        
        # 1st create circle at : 4 -thumb tip and 8 -index finger tip
        x1,y1 = lmList[4][1], lmList[4][2]  # list contains [id, x, y]  - we need (x,y) acc. to id of finger landmark -> 4
        x2, y2 = lmList[8][1], lmList[8][2]  # 8 landmark index finger tip
        cv2.circle(img, (x1,y1), 10, (255,0,255), -1)  # img, center, radius, color, filled '-1' or cv2.FILLED
        cv2.circle(img, (x2,y2), 10, (255,0,255), -1)
        # 2nd create line b/w them
        cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 2)
        # 3rd create center for this line and then put circle for that
        cx, cy= (x1+x2)//2 , (y1+y2)//2
        cv2.circle(img, (cx,cy), 10, (255,0,255), -1)
        # 4th we have to find length b/w "4-8 landmark" tip / length of this line , then we change volume based on that line. This range act like min.-max. volume points.
        length = math.hypot(x2-x1, y2-y1)
        # print(length)    # you can check its goes :  min - 20 , max - 140 , Remember it varies acc. to computer camera process so check length then set min-max
        # 5th Now we convert this min.-max. length acc. to volume range , Hand range : 20 to 140  and  Volume range : -65 to 0
        vol = np.interp(length, [20,140], [minVol, maxVol])    # (total length, [length range min,max], [volume range min,max]) - play with these ranges
        volBar = np.interp(length, [20,140], [400, 150])    # this is for volume bar which display changes acc. to min-max. range in "8th" point below
        volPer = np.interp(length, [20,140], [0, 100])    # for volume percentage  use in "8th" point as text
        # print(int(length), vol)   # print(vol)
        # 6th set master volume level - "from pycaw"
        volume.SetMasterVolumeLevel(vol, None)   # this function set our computer system volume - like : (value : volume)  0 : 100 , -65 : 0

        # 7th Now we create effect like button applying connditions
        if length < 20:
            cv2.circle(img, (cx,cy), 10, (0,255,0), -1)    # if length goes < 15 its center position mark color changes

    # 8th display volume bar - which shows volume changes
    cv2.rectangle(img, (50,150), (80,400), (255,0,0), 2)   # width = 80-50 , height = 400-150,  bcoz we change vol. acc. to 'y' points so vol=0 at 400 , vol=100 at 150 , so we again change our range
    cv2.rectangle(img, (50, int(volBar)), (80,400), (255,0,0), -1)   # this will fill the bar acc. to volume changes, (width will be same, height will be different)
    cv2.putText(img, f"{int(volPer)} %", (40,450), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)    # volume percentage

    # frame rate -fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS : {int(fps)}", (40,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    
    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27:
        break



