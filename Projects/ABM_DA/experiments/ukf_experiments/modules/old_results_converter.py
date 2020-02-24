#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 12:20:53 2020

@author: medrclaa
"""
import os
os.chdir("..")

import sys
import os
import glob




sys.path.append("../../")
from stationsim.ukf2 import pickle_main
sys.path.append("../../../stationsim/")

import ukf2


os.chdir("ukf_old/stationsim/")
import ukf_aggregate
import ukf

os.chdir("../../modules/")
import ukf_fx
import ukf_ex2

os.chdir("..")

"if running this file on its own. this will move cwd up to ukf_experiments."

    
def results_converter(source, prefix, replace = False):
    
    
    """ old results stored as pickled class instances. want to move over to
    pickling class dictionaries instead.
    
    This is much more robust as it doesnt require functions staying the same
    under certain scenarios such as a refactor.
    
    Paramters
    ------
    source : str
        The `source` directory to load and save pickles to/from
        
    replace : bool
    
        If True, replace all files with dictionary versions with the same name.
        Else, return copies of all files with dict_ prefix.
        
    Returns
    -----
    in same folder. returns all files as dictionary pickles instead
    """

    files = glob.glob(source + prefix + "*")
    
    
    
    for file in files:
        file = os.path.split(file)[1]
        u = pickle_main(file, source, True)
        if replace:
                os.remove(source+file)
        else:
            file = "dict_" + file
            
        "check for correct file type ending. old versions have no ending."
        if file[-4:] != ".pkl":
            file += ".pkl"
        u =  pickle_main(file, source, True,instance = u)


if __name__ == "__main__":
    
    source = "/Users/medrclaa/agg_results/"
    prefix = "agg"
    replace = False
    results_converter(source, prefix, replace)