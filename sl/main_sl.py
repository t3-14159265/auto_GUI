# -*- coding: utf-8 -*-



import numpy as np
from utilities import utilities, screenshot, discounted_rewards
from models import model
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import Huber
from actions import actions
from data import target_imagehash, rewards, test_data


#contstants
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BUTTON_NUMBER = 3
SCREEN_COMP_S = 100
ACT_DIM = (SCREEN_WIDTH, SCREEN_HEIGHT, BUTTON_NUMBER)
ENV_DIM = (SCREEN_COMP_S, SCREEN_COMP_S, 3)
t = 10
GAMMA = 0.9
lr = 0.9
next_state, nt_st_hc = None, None
LOAD_WEIGHTS = False
WEIGHTS_NAME = "sl_ckpt"
episodes = 10
beta = 0.0001

#initialise objects
network_cl = model.A2Cnet(ENV_DIM, SCREEN_WIDTH, SCREEN_HEIGHT, LOAD_WEIGHTS, WEIGHTS_NAME)
network = network_cl.get_model()
targ_h_cl = target_imagehash.target_hashcode()
targ_h = targ_h_cl.get_target_hash()
adam_op = Adam(learning_rate=0.00001)
huber_loss = Huber()

log_probs_history = []
action_probs_history = []
critic_value_history = []
reward_history = []
network_opt = RMSprop(lr=lr, epsilon=0.1, rho=0.99)




for episode in range(episodes):
    with tf.GradientTape() as tape:
        #refresh environment
        utilities.refresh_env()
        
        for i in range(t):
            
            #get the state
            if i == 0:
                #select chrome 
                utilities.click_top_of_browser()
                current_state, hash_code = utilities.keras_screenshot()
            else:
                current_state, hash_code = next_state, nt_st_hc 
            current_state = tf.convert_to_tensor(current_state)
            
            #get the action
            width_logits, height_logits, button_logits, val = network(current_state)
            width, height, button = tf.nn.softmax(width_logits), tf.nn.softmax(height_logits), tf.nn.softmax(button_logits)
            width, height, button = utilities.preds_to_numpy(width, height, button)
            val = val[0]
            critic_value_history.append(val)
            
            #get the actions
            action, log_prob, action_probs = actions.move_prob_selection(width, height, button)
            log_probs_history.append(log_prob)
            action_probs_history.append(action_probs)
            
            #execute the action
            utilities.execute_action(action)
            
            #next state
            utilities.click_top_of_browser()
            next_state, nt_st_hc = utilities.keras_screenshot()
            
            #reward
            reward = rewards.get_reward(nt_st_hc, targ_h)
            reward_history.append(reward)
        
        
        #returns
        returns = []
        discounted_sum = 0
        for r in reward_history[::-1]:
            discounted_sum = r + GAMMA * discounted_sum
            returns.insert(0, discounted_sum)
        
        #normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + 0.00001)
        returns = returns.tolist()
        
        #loss
        history = zip(log_probs_history, action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        regularization_loss = 0
        for log_prob, act_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-sum([diff * p for p in log_prob]) + 0.00001)
            critic_losses.append((diff)**2)
            regularization_loss += act_prob * log_prob
        regularization_loss = beta * regularization_loss
        
        
        #backprop
        loss_value = sum(actor_losses) + sum(critic_losses) + regularization_loss
        grads = tape.gradient(loss_value, network.trainable_variables)
        adam_op.apply_gradients(
            (grad, var)
            for (grad, var) in zip(grads, network.trainable_variables)
            if grad is not None)
        
        #save model
        network_cl.upload_model(network)
        network_cl.save_weights()
        
        
        #clear history
        log_probs_history.clear()
        action_probs_history.clear()
        critic_value_history.clear()
        reward_history.clear()
