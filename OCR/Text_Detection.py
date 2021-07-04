# pyttesseract accepts - RGB value ,  Opencv uses BGR , so we convert image into RGB before using tesseract library.
# pip install pytesseract pyocr   # also download exe file , install it and in Path var. define path "TESSDATA_PREFIX" : c:\Program Files\Tesseract-OCR
# Tips improve OCR accuracy : improve quality of image , enhance contrast of image , increasse text size of image , select language at 1 time , remove dark borders / noise , check grammar / spelling


import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'


img = cv2.imread('1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------------------------------------------------------------------------------
# get the text from image
# print(pytesseract.image_to_string(img))    

# ----------------------------------------------------------------------------------------
# Detect character and put bbox around character and label put text on it.
# image_to_boxes() : give the bounding box positions (x,y,w,h) of box for particular character.
# hImg, wImg,_ = img.shape
# boxes = pytesseract.image_to_boxes(img)    # (x,y,w,h) get
# for b in boxes.splitlines():
#     # print(b)
#     b = b.split(' ')  # it will split words based on spaces and we get (x,y,w,h) into list for each character
#     # print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])  # list [character, x,y,w,h] and we will get [1:4] position for each character and all are in string so convert into 'int'
#     cv2.rectangle(img, (x, hImg-y), (w, hImg-h), (255, 255, 255), 1)  # hImg-y : bcoz height is opposite so we have to substarct from Image height
#     cv2.putText(img,b[0],(x, hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 2)

# ------------------------------------------------------------------------------------------
# Detect Word
# boxes = pytesseract.image_to_data(img)      # this will give so many things for particular word so we use slicing -get word
# for a,b in enumerate(boxes.splitlines()):    # gives id for each word
#         # print(b)
#         if a!=0:                  # so 'id' we take if its not equal to 0
#             b = b.split()         # take information into list, at '12' position it holds 'words'
#             # print(b)
#             if len(b)==12:        # so we will take only '12' len of list
#                 x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])       # (6,7,8,9) holds position of (x,y,w,h) in list
#                 cv2.putText(img, b[11], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (50,50,255), 2)
#                 cv2.rectangle(img, (x,y), (x+w, y+h), (50, 50, 255), 1)


# ------------------------------------------------------------------------------------------
# Detect only digits , we use configuration : oem(engine mode) , psm(page segmentation mode) both are in  : '1.png' , '2.png' 
# hImg, wImg,_ = img.shape
# conf = r'--oem 3 --psm 6 outputbase digits'  # default configuration to filter out - only digits, can also use above in 'words'
# boxes = pytesseract.image_to_boxes(img, config=conf)
# for b in boxes.splitlines():
#     print(b)
#     b = b.split(' ')
#     print(b)
#     x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
#     cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
#     cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)



# -----------------------------------------------------------------------------------------------
# store the data in 'txt' or csv file in any format.
from PIL import Image
demo = Image.open('1.png')
text = pytesseract.image_to_string(demo)    # config = tessdata_dir_config

with open('demo.txt', 'w') as f:
    print(text, file=f)




# -----------------------------------------------------------------------------------------------
# Webcam
import numpy as np
from PIL import ImageGrab
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
def captureScreen(bbox=(300,300,1500,1000)):
    capScr = np.array(ImageGrab.grab(bbox))
    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
    return capScr
while True:
    timer = cv2.getTickCount()
    _,img = cap.read()
    #img = captureScreen()
    #DETECTING CHARACTERES
    hImg, wImg,_ = img.shape
    boxes = pytesseract.image_to_boxes(img)     # config = tessdata_dir_config
    for b in boxes.splitlines():
        #print(b)
        b = b.split(' ')
        #print(b)
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img, (x,hImg- y), (w,hImg- h), (50, 50, 255), 2)
        cv2.putText(img,b[0],(x,hImg- y+25),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,255),2)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    #cv2.putText(img, str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,230,20), 2);
    cv2.imshow("Result",img)

    if cv2.waitKey(10) == 27:
        break


# -------------------------------------------------------------------------------------------------
# Webcam / video clip
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'


cap = cv2.VideoCapture(0)             # webcam
# cap = cv2.VideoCapture('clip.mp4')    # clip
# cap.set(cv2.CAP_PROP_FPS, 170)

# if not cap.isOpened():
#     cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     raise IOError('can not open video')

cntr = 0
while True:
    ret, frame = cap.read()
    cntr += 1
    if ((cntr%20) == 0):                   # 20 is no. of frames - to take image of video , change acc.
        imgH, imgW, _ = frame.shape
        x1,y1,w1,h1 = 0,0,imgH, imgW
        imgchar = pytesseract.image_to_string(frame)     # config = tessdata_dir_config
        imgboxes = pytesseract.image_to_boxes(frame)     # config = tessdata_dir_config
        for boxes in imgboxes.splitlines():
            boxes = boxes.split(' ')
            x,y,w,h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
            cv2.rectangle(frame, (x, imgH-y), (w, imgH-h), (0,0,255), 3)  
 
        # cv2.rectangle(frame, (x1,x1), (x1+w1, y1+h1), (0,0,0), -1) 
        cv2.putText(frame, imgchar, (x1 + int(w1/50), y1 +int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)  

        cv2.imshow("Text Detection", frame)
        if cv2.waitKey(1) == 27:
            break


cap.release()
cv2.destroyAllWindows()











# cv2.imshow('image', img)
# cv2.waitKey(0)

