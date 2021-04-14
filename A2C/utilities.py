# -*- coding: utf-8 -*-

import numpy as np
import pyautogui as pg
import pywinauto as pwa
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import time
import tensorflow as tf


def normalise_pixel_data(data):
    data = data / 255
    return data

def keras_screenshot():
    pg.screenshot("screenshot.png")
    
    data = image.load_img("screenshot.png", color_mode="grayscale", target_size=(100, 100))
    data = np.array(data)
    data = normalise_pixel_data(data)
    data = np.expand_dims(data, axis=2)
    data = np.expand_dims(data, axis=0)
    data = np.float32(data)
    
    return data

PAUSE = 1

#############left click#####################
def left_click(wid, height):
    time.sleep(PAUSE)
    print('in the left click method')
    pwa.mouse.click(button='left', coords=(wid, height))

#############right click#####################
def right_click(wid, height):
    time.sleep(PAUSE)
    pwa.mouse.click(button='right', coords=(wid, height))

#############hup and down arrows#################
def up_arrow():
    time.sleep(PAUSE)
    pwa.keyboard.send_keys('{VK_UP}')

def down_arrow():
    time.sleep(PAUSE)
    pwa.keyboard.send_keys('{VK_DOWN}')

#select chrome
def select_chrome():
    left_click(769, 1056)
    time.sleep(0.5)
    left_click(764, 913)





#####################choose move based on probabilities#########
def move_prob_selection(width, height, button):
    wid_idx = list(range(1, len(width)+1))
    height_idx = list(range(1, len(height)+1))
    but_idx = list(range(len(button)))
    
    print(len(width))
    
    wid_choice = np.random.choice(wid_idx, p=width)
    height_choice = np.random.choice(height_idx, p=height)
    but_choice = np.random.choice(but_idx, p=button)
    
    action = (wid_choice, height_choice, but_choice)
    action_probs = (width[wid_choice], height[height_choice], button[but_choice])
    
    return action, action_probs


###################execute action###############################
def execute_action(action):
    
    if action[2] == 0:
        left_click(action[0], action[1])
    if action[2] == 1:
        right_click(action[0], action[1])
    if action[2] == 2:
        up_arrow()
    if action[2] == 3:
        down_arrow()

def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])

def gather_stats(agent, env):
  """ Compute average rewards over 10 episodes
  """
  score = []
  for k in range(10):
      old_state = env.reset()
      cumul_r, done = 0, False
      while not done:
          a = agent.policy_action(old_state)
          old_state, r, done, _ = env.step(a)
          cumul_r += r
      score.append(cumul_r)
  return np.mean(np.array(score)), np.std(np.array(score))

def get_ml_reward():
    rew = input("enter reward")
    return rew