# pip install -q keras-ocr

import keras_ocr
import matplotlib.pyplot as plt

pipeline = keras_ocr.pipeline.Pipeline()

img = "files/1.png"
image = keras_ocr.tools.read(img)

plt.figure(figsize=(10,20))
plt.imshow(image)
