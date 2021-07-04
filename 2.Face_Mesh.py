# 468 points onn face

import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)     # object - find faces
# if want to increase thickness joining points and points on face , otherwise don't use it just comment it out.
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)  # and pass this parameter in  mpDra.draw_landmarks() 2 times - 1 for thickness, 1 for circle radius

cap = cv2.VideoCapture(0)
pTime = 0

while True:
    _, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)    # detect face
    # after detecting, if there are more no. of faces we gonna loop through every face and draw landmarks.
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:    # loop through, faceLms - landmarks of 1 face, we can also add enumerate() here which tell face no. 1 or 2
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawSpec, drawSpec)    # draw landmarks  (img, landmarks, make connection b/w them)

        # to get exact points/limited points on face we gonna number it bcoz there are 468 points, 
        for id, lm in enumerate(faceLms.landmark):    # 1st loop through every landmark and print
            # print(lm)                # gives x,y,z of landmark
            h,w,c = img.shape              # then convert into pixels so use them
            x,y = int(lm.x*w), int(lm.y*h) # pixels values of x,y
            # print(id, x,y)        # we can put them in a list and use acc. we gonna make it into module we create below 2nd part

    # frame rate fps
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)

    cv2.imshow("image", img)
    if cv2.waitKey(1) == 27 :
        break


# -------------------------------------------------------------------------------------------------------------------------------------
# Now craete module :  file  :  "face_mesh_module.py"

class FaceMeshDetector():
    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode= staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)     # find faces
        # if want to increase thickness joining points and points on face , otherwise don't use it just comment it out.
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)  # and pass this parameter in  mpDra.draw_landmarks() 2 times - 1 for thickness, 1 for circle radius


    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)    # detect face
        faces = []    # store every face list
        # after detecting, if there are more no. of faces we gonna loop through every face and draw landmarks.
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:    # loop through, faceLms - landmarks of 1 face, we can also add enumerate() here which tell face no. 1 or 2
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)    # draw landmarks  (img, landmarks, make connection b/w them)

                face=[]    # for 1 face
                # to get exact points/limited points on face we gonna number it bcoz there are 468 points
                # Now we go through - every face for every landmark in face and convert into x,y and then we store into list - for 1 face  and  again store these every face list into another 'faces' list -which holds every face list
                for id, lm in enumerate(faceLms.landmark):    # 1st loop through every landmark and print
                    # print(lm)                # gives x,y,z of landmark
                    h,w,c = img.shape              # then convert into pixels so use them
                    x,y = int(lm.x*w), int(lm.y*h) # pixels values of x,y
                    # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 1)  # show each 'id' of landmark
                    # print(id, x,y)
                    face.append([x,y])
                faces.append(face)

        return img, faces



def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)   # we can pass parameters here

    while True:
        _, img = cap.read()

        img, faces = detector.findFaceMesh(img)    # faces : total no. of faces
        # if len(faces) != 0:
        #     print(len(faces))    # faces[0] -- give list of all points for face 1

        # frame rate fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (20,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 3)

        cv2.imshow("image", img)
        if cv2.waitKey(1) == 27 :
            break



if __name__ == "__main__":
    main()







