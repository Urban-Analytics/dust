#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempt at salvaging old pickles using earlier scipy versions.

@author: medrclaa
"""
import os
os.chdir("..")

import sys
import os
import glob

sys.path.append("../../")
from stationsim.ukf2 import depickler, pickler, class_dict_to_instance
sys.path.append("../../../stationsim/")


"bunch of almost useless imports to keep pickle happy"
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
        u = converter_pickle_main(file, source, True)

        if replace:
                os.remove(source+file)
        else:
            file = "dict_" + file
            
        "check for correct file type ending. old versions have no ending."
        if file[-4:] != ".pkl":
            file += ".pkl"
        u =  converter_pickle_main(file, source, True,instance = u)

def converter_pickle_main(f_name, pickle_source, do_pickle, instance = None):
    
    
    """main function for saving and loading ukf pickles
    
    NOTE THE FOLLOWING IS DEPRECATED IT NOW SAVES AS CLASS_DICT INSTEAD FOR 
    VARIOUS REASONS
    
    - check if we have a finished ukf_ss class and do we want to pickle it
    - if so, pickle it as f_name at pickle_source
    - else, if no ukf_ss class is present, load one with f_name from pickle_source 
        
    IT IS NOW
    
    - check if we have a finished ukf_ss class instance and do we want to pickle it
    - if so, pickle instance.__dict__ as f_name at pickle_source
    - if no ukf_ss class is present, load one with f_name from pickle_source 
    - if the file is a dictionary open it into a class instance for the plots to understand
    - if it is an instance just load it as is.
    
           
    Parameters
    ------
    f_name, pickle_source : str
        `f_name` name of pickle file and `pickle_source` where to load 
        and save pickles from/to
    
    do_pickle : bool
        `do_pickle` do we want to pickle a finished run?
   
    instance : class
        finished ukf_ss class `instance` to pickle. defaults to None 
        such that if no run is available we load a pickle instead.
    """
    
    if do_pickle and instance is not None:
        
        "if given an instance. save it as a class dictionary pickle"
        print(f"Pickling file to {f_name}")
        instance_dict = instance.__dict__
        try:
                """for converting old files. removes deprecated function that
                plays havoc with pickle"""
                instance_dict.pop("ukf")
        except:
            pass
        
        try: 
            "same thing again but for ukf_aggregate"
            instance_dict.pop("ukf_aggregate")

        except:
            pass
        
        pickler(instance.__dict__, pickle_source, f_name)
        return
    
    else:
        file = depickler(pickle_source, f_name)
        print(f"Loading pickle {f_name}")
        "try loading the specified file as a class dict. else an instance."
        if type(file) == dict:
            "removes old ukf function in memory"
            
            
            instance =  class_dict_to_instance(file)
        else: 
            instance = file
            
        return instance
 
    
if __name__ == "__main__":
    
    source = "/Users/medrclaa/agg_results/"
    prefix = "agg"
    replace = False
    results_converter(source, prefix, replace)