# -*- coding: utf-8 -*-

import numpy as np

def discount(r, gamma):
    """ Compute the gamma-discounted rewards over an episode
    """
    discounted_r, cumul_r = np.zeros_like(r), 0
    for t in reversed(range(0, len(r))):
        cumul_r = r[t] + cumul_r * gamma
        discounted_r[t] = cumul_r
    return discounted_r