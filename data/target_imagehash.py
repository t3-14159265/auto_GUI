# -*- coding: utf-8 -*-

from utilities import utilities
import os
import pickle

def create_target_hash():
    utilities.select_chrome()
    utilities.click_top_of_browser()
    data, hashcode = utilities.keras_screenshot()
    return hashcode

class target_hashcode:
    def __init__(self):
        if os.path.isfile("targ_h.pickle"):
            print("loading existing target hascode")
            file_in = open("targ_h.pickle", "rb")
            self.targ_h = pickle.load(file_in)
            file_in.close()
        else:
            self.targ_h = create_target_hash()
            self.save_tar_hash()
    
    def save_tar_hash(self):
        file_out = open("targ_h.pickle", "wb")
        pickle.dump(self.targ_h, file_out)
        file_out.close()
    
    def get_target_hash(self):
        return self.targ_h

if __name__ == "__main__":
    targ_h_cl = target_hashcode()