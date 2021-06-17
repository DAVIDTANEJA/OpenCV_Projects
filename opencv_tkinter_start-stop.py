from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import cv2, imutils


def visualizer():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            frame = imutils.resize(frame, width=640)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            lblVideo.configure(image=img)
            lblVideo.image = img
            lblVideo.after(10, visualizer)
        else:
            lblVideo.image = ""
            cap.release()


def Start_webcam():
    global cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    visualizer()

def Stop_webcam():
    global cap
    cap.release()


cap = None
root = Tk()

btnStart = Button(root, text="Start Webcam", width=45, command=Start_webcam)
btnStart.grid(column=0, row=0, padx=5, pady=5)

btnStop = Button(root, text="Stop", width=45, command=Stop_webcam)
btnStop.grid(column=1, row=0, padx=5, pady=5)

lblVideo = Label(root)
lblVideo.grid(column=0, row=1, columnspan=2)


root.title('Webcam')
root.mainloop()