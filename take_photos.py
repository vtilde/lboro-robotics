import ipywidgets.widgets as widgets
from IPython.display import display
image_display = widgets.Image(format="jpeg")
# display(image_display)

import cv2
import numpy as np
import pyzed.sl as sl

def matrix_to_jpeg(value):
    return bytes(cv2.imencode('.jpg',value)[1])
    
camera = sl.Camera()

camera_params = sl.InitParameters()
camera_params.camera_resolution = sl.RESOLUTION.HD720
camera_params.depth_mode = sl.DEPTH_MODE.ULTRA
camera_params.coordinate_units = sl.UNIT.MILLIMETER

status = camera.open(camera_params)
if status != sl.ERROR_CODE.SUCCESS:
    print("camera error")
    print(status)
    exit()

def take_photo(path):
    err = camera.grab()
    if err == sl.ERROR_CODE.SUCCESS:
        image_data = sl.Mat()
        camera.retrieve_image(image_data)
        image = image_data.get_data()
        # image_display.value = matrix_to_jpeg(image)
        cv2.imwrite(path, image)

        
take_photo("photos/line_image_" + str(cv2.getTickCount()) + ".jpg")
