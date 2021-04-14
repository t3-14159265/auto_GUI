# -*- coding: utf-8 -*-
from utilities import utilities
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


class test_ep_data:
    def __init__(self):
        if os.path.isfile("episode.pickle"):
            file_in = open("episode.pickle", "rb")
            self.episode_data = pickle.load(file_in)
            file_in.close()
        else:
            self.episode_data = []
    
    def save_data(self):
        file_out = open("episode.pickle", "wb")
        pickle.dump(self.episode_data, file_out)
        file_out.close()
    
    def add_to_data(self, data):
        for data_point in data:
            self.episode_data.append(data_point)
            self.save_data()
    
    def get_data(self):
        return self.episode_data



if __name__ == "__main__":
    i = 0
    data_cl = test_ep_data()
    data = data_cl.get_data()
    example = np.array(data[i][0])
    example = np.squeeze(example, 0)
    plt.imshow(example)
    print(data[i][1])
    print(data[i][2])
    data_cl.save_data()