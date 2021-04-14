# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers


##############4 HEADED MODEL#################################
class A2Cnet:
    def __init__(self, input_shape, screen_width, screen_height, model_load_w):
        self.input_shape = input_shape
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.l2_reg = 1e-5
        self.model_load_w = model_load_w
        self.build_model()
        
        
    
    def build_model(self):
        #input
        self.img_inputs = layers.Input(shape=self.input_shape)
        #body of model
        self.image_net = layers.Conv2D(filters=16, kernel_size=5, strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg), bias_regularizer=regularizers.l2(self.l2_reg))(self.img_inputs)
        self.image_net = layers.Conv2D(filters=32, kernel_size=3, strides=(1,1), padding="SAME", activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg), bias_regularizer=regularizers.l2(self.l2_reg))(self.image_net)
        self.image_dense = layers.Flatten()(self.image_net)
        self.image_dense = layers.Dense(units=500, activation="relu", kernel_regularizer=regularizers.l2(self.l2_reg), bias_regularizer=regularizers.l2(self.l2_reg))(self.image_dense)
        #outputs
        self.width_header = layers.Dense(units=self.screen_width, activation="linear", kernel_regularizer=regularizers.l2(self.l2_reg), bias_regularizer=regularizers.l2(self.l2_reg))(self.image_dense)
        self.height_header = layers.Dense(units=self.screen_height, activation="linear", kernel_regularizer=regularizers.l2(self.l2_reg), bias_regularizer=regularizers.l2(self.l2_reg))(self.image_dense)
        self.button_header = layers.Dense(units=3, activation="softmax", name="linear", kernel_regularizer=regularizers.l2(self.l2_reg), bias_regularizer=regularizers.l2(self.l2_reg))(self.image_dense)
        self.value_header = layers.Dense(units=1, activation="linear", name="value", kernel_regularizer=regularizers.l2(self.l2_reg), bias_regularizer=regularizers.l2(self.l2_reg))(self.image_dense)
        
        #model
        self.model = keras.Model(inputs=[self.img_inputs], outputs=[self.width_header, self.height_header, self.button_header, self.value_header])
        if self.model_load_w:    
            self.model.load_weights("ckpt")
        
    def get_model(self):
        return self.model
    
    def single_prediction(self, input_data):
        return self.model(input_data)
    
    def fit_model(self, x_train, y_train):
        self.history = self.model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=2,
            validation_split=0.1
            )
    
    
    
    def upload_model(self, model):
        self.model = model
    
    def save_weights(self):
        self.model.save_weights("ckpt")