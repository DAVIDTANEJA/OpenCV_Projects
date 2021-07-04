# getting 6 points on face.
# mediapipe takes RGB image, so convert it into while loop befor use.
# Remember : in webcam we have limited no. of fps , but in video clips which are faster no. of fps are more.
# To draw rectangle on face : 1.we can use mediapipe provided function : mpDraw.draw_detection(img, detection)  or  2.we use  "bbox" after converting x,y,width,height  into 'pixel' values and draw rectangle on faces.

import cv2
import mediapipe as mp
import time


mpFaceDetection = mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection()   # 0.75 - can increase min face detection confidnce
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
pTime=0

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # RGB
    results = faceDetection.process(imgRGB)
    # print(results)

    # give the details of faces with particular id , details like : score, location_data -positions of x,y acc. to faces
    if results.detections:
        for id, detection in enumerate(results.detections):
            # x,y values are normalized values b/w 0-1 we have to multiply with width, height to get 'pixel' values so we can draw
            # we can also use mediapipe provided function : mpDraw.draw_detection(img, detection)  or we use  "bbox" after converting x,y,width,height  into 'pixel' values and draw rectangle on faces.
            # mpDraw.draw_detection(img, detection)    # 1.mediapipe method - draw rectangle face
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)   # x,y positions
            # to get information for xmin, ymin we have to write so long : detection.location_data.relative_bounding_box.xmin , instead of this we store in var.
            bboxC = detection.location_data.relative_bounding_box   # bounding box class - bboxC and later convert into pixels acc.
            h,w,c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)   # xmin, ymin, width, height  in 'pixels' - use these to draw rectangle on faces
            cv2.rectangle(img, bbox, (255,0,255), 2)     # 2.using - bbox to draw rectangle on face
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2)   # put score on face

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'fps : {int(fps)}', (20,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) == 27:
        break


# -----------------------------------------------------------------------------------------------------------------
# Now create module  :   file : "face_detection_module.py"

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)   # 0.75 - can increase min face detection confidnce
        self.mpDraw = mp.solutions.drawing_utils

    # find faces and return : bbox, id, score - in a list for every face
    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # RGB
        self.results = faceDetection.process(imgRGB)
        # print(self.results)
        bboxes = []
        # give the details of faces with particular id , details like : score, location_data -positions of x,y acc. to faces
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box   # bounding box class - bboxC and later convert into pixels acc.
                h,w,c = img.shape
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)   # xmin, ymin, width, height  in 'pixels' - use these to draw rectangle on faces
                bboxes.append([id, bbox, detection.score])
                
                if draw:     # True
                    img = self.fancyDraw(img, bbox)  # fancy draw rectangle box

                    # cv2.rectangle(img, bbox, (255,0,255), 2)     # 2.using - bbox to draw rectangle on face ,   comment this out bcoz using  -- fancyDraw()
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255), 2)   # put score on face

        return img, bboxes    # bboxes -list of : bbox, id, score

    # using this to draw fancy rectangle box in above method
    def fancyDraw(self, img, bbox, l=30, rt=1, t=5):    # l- length of line, rt -rectangle thickness, t-thickness for x,y
        x, y, w, h = bbox    # x,y - origin points
        x1, y1 = x+w, y+h    # x1,y1 - diagonal points
        
        cv2.rectangle(img, bbox, (255,0,255), rt)        # 2.using - bbox to draw rectangle on face
        # Top left x,y point
        cv2.line(img, (x,y), (x+l, y), (255,0,255), t)   # img, starting point, length of line, color, thickness
        cv2.line(img, (x,y), (x, y+l), (255,0,255), t)   # img, starting point, length of line, color, thickness
        # Top right x1,y point
        cv2.line(img, (x1,y), (x1-l, y), (255,0,255), t)   # img, starting point, length of line, color, thickness
        cv2.line(img, (x1,y), (x1, y+l), (255,0,255), t)   # img, starting point, length of line, color, thickness
        # bottom left x,y1 point
        cv2.line(img, (x,y1), (x+l, y1), (255,0,255), t)   # img, starting point, length of line, color, thickness
        cv2.line(img, (x,y1), (x, y1-l), (255,0,255), t)   # img, starting point, length of line, color, thickness
        # bottom right x1,y1 point
        cv2.line(img, (x1,y1), (x1-l, y1), (255,0,255), t)   # img, starting point, length of line, color, thickness
        cv2.line(img, (x1,y1), (x1, y1-l), (255,0,255), t)   # img, starting point, length of line, color, thickness

        return img


def main():
    cap = cv2.VideoCapture(0)
    pTime=0
    detector = FaceDetector()    # object for class

    while True:
        _, img = cap.read()
        img, bboxes = detector.findFaces(img)   # findFaces() method - this methods returns 2 values : img, bboxes
        # print(bboxes)

        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'fps : {int(fps)}', (20,70), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) == 27:
            break




if __name__ == "__main__":
    main()

