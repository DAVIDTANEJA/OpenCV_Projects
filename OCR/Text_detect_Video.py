# Break the video clips into frames and read those frames using pytesseract
import cv2
import os
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
tessdata_dir_config = '--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata"'

# # create folder to store frames 'image_frames'
# if not os.path.exists("image_frames"):
#     os.makedirs("image_frames")


# test_vid = cv2.VideoCapture('clip.mp4')    # video clip


# count = 0   # count for the frames
# while test_vid.isOpened():
#     ret, frame = test_vid.read()
#     if not ret:
#         break

#     # name for frames file
#     name = "image_frames/frame" + str(count) + ".png"

#     cv2.imwrite(name, frame)
#     count += 1

#     # break the loop acc.
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# test_vid.release()
# cv2.destroyAllWindows()

# test the frame image
demo = Image.open("image_frames/frame61.png")
# demo = cv2.cvtColor(demo, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(demo, lang='eng', config = tessdata_dir_config)
print(text)


# -------------------------------------------------------------------------
# press 's' to save frames from video
camera=cv2.VideoCapture(0)

while True:
    _,image=camera.read()
    cv2.imshow('image',image)
    if cv2.waitKey(1)& 0xFF==ord('s'):
        cv2.imwrite('test1.jpg',image)
        break
camera.release()
cv2.destroyAllWindows()

def tesseract():
    path_to_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image_path = "test1.jpg"
    pytesseract.tesseract_cmd = path_to_tesseract
    text = pytesseract.image_to_string(Image.open(image_path))
    print(text[:-1])
tesseract()