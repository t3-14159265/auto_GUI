# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras.optimizers import RMSprop

class Agent:
    #generic agent class
    def __init__(self, inp_dim, out_dim, lr):
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.rms_optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)
    
    def fit(self, inp, targ):
        # perform one epoch of training
        self.model.fit(self.reshape(inp), targ, epochs=1, verbose=0)
    
    def predict(self, inp):
        #critic value prediction
        return self.model.predict(self.reshape(inp))
    
    
    def reshape(self, x):
        if len(x.shape) < 3: return np.expand_dims(x, axis=0)
        else: return x
