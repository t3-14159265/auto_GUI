# -*- coding: utf-8 -*-

import pywinauto as pwa
import pyautogui as pg
import time
import numpy as np
from utilities import utilities



#####################choose move based on probabilities#########
def move_prob_selection(width, height, button):
    wid_idx = list(range(len(width)))
    height_idx = list(range(len(height)))
    but_idx = list(range(len(button)))
    
    print(len(width))
    
    wid_choice = np.random.choice(wid_idx, p=width)
    height_choice = np.random.choice(height_idx, p=height)
    but_choice = np.random.choice(but_idx, p=button)
    
    action = (wid_choice, height_choice, but_choice)
    
    ##############this should be log action probs
    
    action_probs = [width[wid_choice], height[height_choice], button[but_choice]]
    log_probs = np.log(action_probs)
    return action, log_probs


###################execute action###############################
def execute_action(action):
    utilities.click_top_of_browser()
    if action[2] == 0:
        utilities.left_click(action[0], action[1])
    if action[2] == 1:
        utilities.up_arrow()
    if action[2] == 2:
        utilities.down_arrow()
