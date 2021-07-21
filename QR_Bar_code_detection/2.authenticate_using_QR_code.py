# we have list of approved Ids in a file "data.txt" and whenever person comes and shows ID the system will check person authorized or not.
# we can also use this as an attendance system.

import cv2
import numpy as np
from pyzbar.pyzbar import decode

#img = cv2.imread('1.png')
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

# read the data and store in a list
with open("files/data.txt") as f:
    myDataList = f.read().splitlines()    # read all the data and based on the lines add 1 item to list.
# print(myDataList)


while True:
    success, img = cap.read()
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')    # 1.decode the data / read image
        print(myData)

        if myData in myDataList:                 # 2.check if person is authorised/not  in data list
            myOutput = 'Authorized'
            myColor = (0,255,0)
        else:
            myOutput = 'Un-Authorized'
            myColor = (0, 0, 255)

        # 3.we will draw polygon around barcode, polygon bcoz if we rotate image it capture also
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,myColor,5)
        pts2 = barcode.rect
        cv2.putText(img, myOutput, (pts2[0],pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)  # also put text 'output'

    cv2.imshow('Result',img)
    cv2.waitKey(1)
