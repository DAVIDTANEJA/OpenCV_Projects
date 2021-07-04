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
            lblInfoVideoPath.configure(text="Selecting File")
            lblVideo.image = ""
            cap.release()


def elegir_visulaizer_video():
    global cap
    video_path = filedialog.askopenfilename(filetypes=[("all video format",".mp4"), ("all video format",".avi")])
    if len(video_path)>0:
        lblInfoVideoPath.configure(text=video_path)
        cap = cv2.VideoCapture(video_path)
        visualizer()
    else:
        lblInfoVideoPath.configure(text="Selecting File")


cap = None
root = Tk()

btnVisualizer = Button(root, text="Browse to visualize Video", command=elegir_visulaizer_video)
btnVisualizer.grid(column=0, row=0, padx=5, pady=5, columnspan=2)

lblInfo1 = Label(root, text='Video File location:')
lblInfo1.grid(column=0, row=1)

lblInfoVideoPath = Label(root, text="Here shows File path location")
lblInfoVideoPath.grid(column=1, row=1)

lblVideo = Label(root)
lblVideo.grid(column=0, row=2, columnspan=2)

root.mainloop()


