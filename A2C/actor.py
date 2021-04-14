# -*- coding: utf-8 -*-

import numpy as np
import tensorflow.keras.backend as K

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from .agent import Agent

class Actor(Agent):
    """ Actor for the A2C Algorithm
    """

    def __init__(self, inp_dim, out_dim, network, lr):
        Agent.__init__(self, inp_dim, out_dim, lr)
        self.model = self.addHead(network)
        self.action_pl_w = K.placeholder(shape=(None, self.out_dim[0]))
        self.action_pl_h = K.placeholder(shape=(None, self.out_dim[1]))
        self.action_pl_b = K.placeholder(shape=(None, self.out_dim[2]))
        self.action_pl = K.placeholder(shape=(None, self.out_dim[0], self.out_dim[1], self.out_dim[2]))
        self.advantages_pl = K.placeholder(shape=(None,))
    def addHead(self, network):
        """ Assemble Actor network to predict probability of each action
        """
        x = Dense(128, activation='relu')(network.output)
        out_width = Dense(self.out_dim[0], activation='softmax')(x)
        out_height = Dense(self.out_dim[1], activation='softmax')(x)
        out_button = Dense(self.out_dim[2], activation='softmax')(x)
        
        return Model(inputs=network.input, outputs=[out_width, out_height, out_button])

    def optimizer(self):
        """ Actor Optimization: Advantages + Entropy term to encourage exploration
        (Cf. https://arxiv.org/abs/1602.01783)
        """
        weighted_actions_w = K.sum(self.action_pl_w * self.model.output[0], axis=1)
        weighted_actions_h = K.sum(self.action_pl_h * self.model.output[1], axis=1)
        weighted_actions_b = K.sum(self.action_pl_b * self.model.output[2], axis=1)
        weighted_actions = weighted_actions_w + weighted_actions_h + weighted_actions_b
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy_w = K.sum(self.model.output[0] * K.log(self.model.output[0] + 1e-10), axis=1)
        entropy_h = K.sum(self.model.output[1] * K.log(self.model.output[1] + 1e-10), axis=1)
        entropy_b = K.sum(self.model.output[2] * K.log(self.model.output[2] + 1e-10), axis=1)
        entropy = entropy_w + entropy_h + entropy_b
        loss = 0.001 * entropy - K.sum(eligibility)
        
        updates = self.rms_optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def save(self, path):
        self.model.save_weights(path + '_actor.h5')

    def load_weights(self, path):
        self.model.load_weights(path)