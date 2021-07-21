# It works for both QR ad Bar code.
import cv2
import numpy as np
from pyzbar.pyzbar import decode  # helps in detect and their position (x,y,w,h) of QR and Bar code, decode message as well.

# img = cv2.imread('1.png')
cap = cv2.VideoCapture(0)      # using webcam
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    for barcode in decode(img):                # now 'barcode' has all the data, rect, polygon, other information
        myData = barcode.data.decode('utf-8')  # barcode.data - is in binary/bytes so decode it(can be number/text/anything). 
        print(myData)
        # now we will draw polygon around barcode , polygon bcoz if we rotate image it capture also
        pts = np.array([barcode.polygon], np.int32)        # 1st convert into array
        pts = pts.reshape((-1,1,2))                        # then reshape the array
        cv2.polylines(img, [pts], True, (255,0,255), 5)    # then draw using cv2.polylines(img, points, closed -True, color, thickness)
        # using rectangle to put text, if we use polygon and we rotate img it will have angle and text also rotate, so using rectangle
        pts2 = barcode.rect
        cv2.putText(img, myData, (pts2[0],pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)  # put text data

    cv2.imshow('Result',img)
    if cv2.waitKey(1) == 27:
        break
