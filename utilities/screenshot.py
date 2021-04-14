# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:50:05 2020

@author: thoma
"""

import numpy as np
import pyautogui as pg
import pywinauto as pwa
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image


SCREEN_WIDTH = int(1920)
SCREEN_HEIGHT = int(1080)


def normalise_pixel_data(data):
    data = data / 255
    return data

def keras_screenshot():
    pg.screenshot("screenshot.png")
    
    data = image.load_img("screenshot.png", color_mode="rgb", target_size=(100, 100))
    data = np.array(data)
    data = normalise_pixel_data(data)
    #data = np.expand_dims(data, axis=2)
    data = np.expand_dims(data, axis=0)
    data = np.float32(data)
    
    return data


