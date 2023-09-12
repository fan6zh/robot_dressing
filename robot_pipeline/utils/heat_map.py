import sys

sys.dont_write_bytecode = True
import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
import cv2


def close_event():
    plt.close()  # timer calls this function after 3 seconds and closes the window


def add(image, heat_map, alpha=0.6, display=False, save=None, cmap='plasma', axis='on', verbose=False):
    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    fig, ax = plt.subplots()
    timer = fig.canvas.new_timer(interval=3000)  # creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax.imshow(image)
    ax.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    # timer.start()
    plt.show()
