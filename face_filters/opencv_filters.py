import cv2
from PIL import Image
import numpy as np

faceCascade = cv2.CascadeClassifier("../OpenCV/haarcascade/haarcascade_frontal_face_default.xml")

maskPath = "thug_life.png"
mask = Image.open(maskPath)

# detect face and put mask filter image on face also resize acc. to face
def thug_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 2.1)
    background = Image.fromarray(image)        # put mask image as an array on face.
    for (x,y,w,h) in faces:
        resized_mask = mask.resize((w,h), Image.ANTIALIAS)    # resize mask image acc. to face
        offset = (x,y)
        background.paste(resized_mask, offset, mask=resized_mask)    # paste resize_mask image on background real face

    return np.asarray(background)

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    cv2.imshow("Thug life filter", thug_mask(frame))

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()