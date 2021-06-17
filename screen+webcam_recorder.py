import cv2
import numpy as np
from PIL import ImageGrab
import datetime, keyboard

SCREEN_SIZE = (1366, 768)          # size=(ImageGrab.grab()).size  - give size of screen 

webcam = cv2.VideoCapture(0)    # 0 -default camera , 1 -for other camera, also can pass videofile here 'video1.mp4'
# cv2.resizeWindow('Capture', 100, 100)

def screenrecorder():
    # filename = f"{datetime.datetime.now().strftime('%d-%m-%Y, %I-%M-%S %p')}.mp4"  # dynamic filename acc. to time format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # codec : 'XVID' -.avi, 'mp4v', 'X264', 'DIVX', 'MJPG'
    fps = 10.0
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (SCREEN_SIZE))

    while True:
        img = ImageGrab.grab()  # take screenshot ,  (bbox=(0,0,500,500)) - for particular dimension to record, if using 'bbox' then don't use 'SCREEN_SIZE'
        np_img = np.array(img)
        final_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)     # convert images into RGB, 'final_imag' - is final window captured
        
        # webcam.set(cv2.CAP_PROP_FRAME_WIDTH,50)      # set width, height of webcam
        # webcam.set(cv2.CAP_PROP_FRAME_HEIGHT,50)
        _, frame = webcam.read()                        # capture video frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print('frame', frame.shape)
        print('final image', final_img.shape)
        fr_height, fr_width, _ = frame.shape
        smaller_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)  # (240,320,3)(height,width)
        final_img[510:750, 1030:1350] = smaller_frame  # here put our webcam frame onto final window 'final_img'

        out.write(final_img)

        if keyboard.is_pressed("esc"):            # "q"
            break

    out.release()
    webcam.release()
    cv2.destroyAllWindows()


# call the function
screenrecorder()


