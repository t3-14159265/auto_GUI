# -*- coding: utf-8 -*-

import imagehash
import numpy as np

def get_reward(hash_c, targ_c):
    reward = 3 / np.abs(targ_c - hash_c) + 0.00001
    return reward