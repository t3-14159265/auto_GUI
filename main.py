# -*- coding: utf-8 -*-



import numpy as np
from utilities import utilities, screenshot, discounted_rewards
from models import model
from tensorflow.keras.optimizers import RMSprop, Adam
from actions import actions

#contstants
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BUTTON_NUMBER = 4
SCREEN_COMP_S = 100
ACT_DIM = (SCREEN_WIDTH, SCREEN_HEIGHT, BUTTON_NUMBER)
ENV_DIM = (SCREEN_COMP_S, SCREEN_COMP_S, 3)
t = 20
GAMMA = 0.9
lr = 0.9

#initialise objects
network = model.A2C(ENV_DIM, SCREEN_WIDTH, SCREEN_HEIGHT)
adam_op = Adam(learning_rate=0.01)
state_history = []
action_probs_history = []
reward_history = []
network_opt = RMSprop(lr=lr, epsilon=0.1, rho=0.99)

#play k steps of episode
for i in range(t):
    
    #select chrome
    utilities.select_chrome()
    
    #get the state
    current_state = screenshot.keras_screenshot()
    state_history.append(current_state)
    
    #get the action
    width, height, button, val = network.single_prediction(current_state)
    
    
    #action_probs_history.append()
    action, action_prob = actions.move_prob_selection(width, height, button)
    action_probs_history.append(action_prob)
    
    
    #execute the action
    utilities.execute_action(action)
    
    #reward
    reward = utilities.get_ml_reward()
    reward_history.append(reward)




#discounted reward totals
dis_rewards = discounted_rewards(reward_history, GAMMA)

history = list(zip(state_history, action_probs_history, dis_rewards))

#loss

