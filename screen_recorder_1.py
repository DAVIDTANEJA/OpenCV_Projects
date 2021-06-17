# create program - which take screenshots repeatedly and write those into a video file.

# Screen Sizes :
# Enhanced-definition television (EDTV):
    # 480p (720 × 480 progressive scan)   # 576p (720 × 576 progressive scan)
# High-definition television (HDTV):
    # 720p (1280 × 720 progressive scan)  # 1080i (1920 × 1080 split into two interlaced fields of 540 lines) 
    # 1080p (1920 × 1080 progressive scan)
# Ultra-high-definition television (UHDTV):
    # 4K UHD (3840 × 2160 progressive scan)  # 8K UHD (7680 × 4320 progressive scan)

import cv2, time
import numpy as np
import pyautogui    # take screenshot , we can also use : from PIL import ImageGrab

# if error in running output file , check your PC display resolution use that "Settings-> System-> Display-> Display Resolution"
SCREEN_SIZE = (1366, 768)

# codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")     # "mp4v"
# video writer
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (SCREEN_SIZE))    # (filename, codec, fps, screen size)
fps = 120
prev = 0       # slow down video, increase value decrease video speed


# 1st take screenshot , 2nd into numpy array, 3rd into image and video using cv2, finally write output.
while True:
    time_elapsed = time.time() - prev
    img = pyautogui.screenshot()            # store the screenshot in pixel
    if time_elapsed > 1.0/fps:
        prev = time.time()           # to update time
        frame = np.array(img)        # convert 'img' into numpy array
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # create image
        out.write(frame)    # create output

    cv2.waitKey(100)             # wait for 100 ms-'mili sec.'

cv2.destroyAllWindows()
out.release()
