#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:51:00 2019

@author: rob

generalised arc script

Run some experiments using the stationsim/particle_filter.
The script
@author: RC

run following in bash console:
    
#log in to arc (requires UoL login)
ssh medrclaa@arc3.leeds.ac.uk
#move to nobackup storage
cd /nobackup/medrclaa
#clone in git repository
#git clone https://github.com/Urban-Analytics/dust/
git clone -b dev_rc https://github.com/Urban-Analytics/dust/

#move to experiments folder
cd dust/Projects/ABM_DA/experiments/ukf_experiments

#create python 3 virtual environment

# ffmpeg-imageio not on pip. eith
module load python python-libs
virtualenv mypython
source mypython/bin/activate


#install in dependencies.
# for filtering
pip install filterpy
#for polygons
pip install shapely
# for plotting
pip install seaborn
# for animations
pip install imageio
 
# also needs imageio-ffmpeg from conda forge. 
# idk if theres a pip alternative but doesnt matter for arc.

#to run this file (arc.py)
qsub arc.sh

when exporting from arc in another linux terminal use 

scp username@leeds.ac.uk:source_in_arc/* destination_in_linux/.

e.g.

from linux terminal 
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/ukf* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
"""

import sys
import os

if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")

sys.path.append('../../stationsim')
from ukf2 import ukf_ss, pickler
from stationsim_model import Model

from modules.ukf_ex0 import ex0_params, ex0_save
from modules.ukf_ex1 import omission_params
from modules.ukf_ex2 import aggregate_params
from modules import default_ukf_configs as configs

import numpy as np
from os import remove

#%%
def ex0_input(model_params,ukf_params, test):
    
    """update model and ukf dictionaries so experiment 0 runs
    
    - define some population, list of noises and list of sampling rates.
    - define a range of repetitions (run_id) for each noise/rate pair.
    - put the above into one large parameter list so each item has one of the unique
        noise/rate/run_id products.
    - we then choose one item of this list using the arc task_id as our list of parameters
        for this run
    - using these parameters update model_params and ukf_params using ex0_params
    - add `run_id`, `f_name`, and `do_pickle` required for arc run.
    - output updated dictionaries
    
    Parameters
    ------
    
    model_params, ukf_params : dict
        dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf.


    Returns
    ------
    
    model_params, ukf_params : dict
        updated dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf. 
    
    base_model : cls
        `base_model` initialised stationsim instance

    """
    
    
    n = 5 # 10 to 30 agent population by 10
    
    sample_rate = [1,2,5,10] # how often to assimilate with ukf
    noise = [0,0.25,0.5,1,2,5] #list of gaussian noise standard deviations
    run_id = np.arange(0,30,1)  # 30 runs

    param_list = [(x, y, z) for x in sample_rate for y in noise for z in run_id]
    
    
    if not test:
        sample_rate = param_list[int(sys.argv[1])-1][0]
        noise = param_list[int(sys.argv[1])-1][1]
        run_id = param_list[int(sys.argv[1])-1][2]
        
    else:
        sample_rate = 10
        noise = 5
        run_id = "test"

    model_params, ukf_params, base_model = ex0_params(n, noise, sample_rate,
                                                      model_params, ukf_params)
    
    ukf_params["run_id"] = run_id

    ukf_params["f_name"] = "config_agents_{}_rate_{}_noise_{}-{}".format(      
                            str(n).zfill(3),
                            str(float(sample_rate)),
                            str(float(noise)),
                            str(run_id).zfill(3))
    
    ukf_params["file_name"] = ukf_params["f_name"] + ".npy"
    ukf_params["do_pickle"] = False
    
    return model_params, ukf_params, base_model
    
def ex1_input(model_params,ukf_params, test):
    
    """update model and ukf dictionaries so experiment 1 runs
    
    - define some population, list of populations and proportions observed.
    - define a range of repetitions (run_id) for each pop/prop pair.
    - put the above into one large parameter list so each item has one of the unique
        noise/rate/run_id products.
    - we then choose one item of this list using the arc task_id as our list of parameters
        for this run
    - using these parameters update model_params and ukf_params using ex0_params
    - add `run_id`, `f_name`, and `do_pickle` required for arc run.
    - output updated dictionaries
    
    Parameters
    ------
    
    model_params, ukf_params : dict
        dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf.

    Parameters
    ------
    
    model_params, ukf_params : dict
        updated dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf. 

    """
    
    
    num_age = [10,20,30]  # 10 to 30 agent population by 10
    props = [0.25,0.5,0.75,1]  # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
    run_id = np.arange(0,30,1)  # 30 runs
    
    param_list = [(x, y,z) for x in num_age for y in props for z in run_id]
    
    if not test:
        n =  param_list[int(sys.argv[1])-1][0]
        prop = param_list[int(sys.argv[1])-1][1]
        run_id = param_list[int(sys.argv[1])-1][2]
    else:
        n = 5
        prop = 0.5
        run_id = "test"        
        
    model_params, ukf_params, base_model = omission_params(n, prop, 
                                                           model_params, ukf_params)
    
    ukf_params["run_id"] = run_id
    ukf_params["file_name"] = "ukf_agents_{}_prop_{}-{}".format(      
                            str(n).zfill(3),
                            str(prop),
                            str(run_id).zfill(3))+ ".pkl"
    
    ukf_params["do_pickle"] = True
    
    return model_params, ukf_params, base_model

def ex2_input(model_params, ukf_params, test):
    
    """update model and ukf dictionaries so experiment 2 runs
    
    - define some population, list of noises and list of aggregate squares `bin_size`.
    - define a range of repetitions (run_id) for each pop/bin_size pair.
    - put the above into one large parameter list so each item has one of the unique
        noise/rate/run_id products.
    - we then choose one item of this list using the arc task_id as our list of parameters
        for this run
    - using these parameters update model_params and ukf_params using ex0_params
    - add `run_id`, `f_name`, and `do_pickle` required for arc run.
    - output updated dictionaries
    
    Parameters
    ------
    
    model_params, ukf_params : dict
        dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf.

    Parameters
    ------
    
    model_params, ukf_params : dict
        updated dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf. 

    """
    
    num_age = [10,20,30]  # 10 to 30 agent population by 10
    bin_size = [5,10,25,50]  # unitless grid square size (must be a factor of 100 and 200)
    run_id = np.arange(0,30,1)  # 30 runs
    
    param_list = [(x, y,z) for x in num_age for y in bin_size for z in run_id]
    
    if not test:
        n = param_list[int(sys.argv[1])-1][0] 
        bin_size = param_list[int(sys.argv[1])-1][1]
        run_id = param_list[int(sys.argv[1])-1][2]
    
    else:
        n = 5
        bin_size = 50
        run_id = "test2"
        
    model_params, ukf_params, base_model = aggregate_params(n, bin_size,
                                                            model_params, ukf_params)
    
    ukf_params["run_id"] = run_id
    ukf_params["pickle_file_name"] = "agg_ukf_agents_{}_bin_{}-{}".format(      
                            str(n).zfill(3),
                            str(bin_size),
                            str(run_id).zfill(3)) + ".pkl"
    ukf_params["full_file_name"] = ukf_params["pickle_file_name"]
   
    ukf_params["do_pickle"] = True

    return model_params, ukf_params, base_model



def main(ex_input,ex_save=None, test = False):
    
    
    """main function for running ukf experiments in arc.
    
    Runs an instance of ukf_ss for some experiment and associated parameters.
    Either pickles the run or saves some user defined aspect of said run.
    
    - define some input experiment for arc
    - update model_params and ukf_params dictionaries for said experiment
    - run ukf_ss with said parameters
    - pickle or save metrics when run finishes.
    
    Parameters
    ------
    ex_input : func
        `ex_input` some experiment input that updates the default model_params
        and ukf_params dicitonaries with items needed for filter to run given
        experiment.
        
    ex_save : func
        `ex_save` if we want to pickle the whole class instance set this to none.
        Else provides some alternate function that saves something else. 
        Experiment 0 has a funciton ex0 save which saves a numpy array of 
        error metric instead
    
    """

    if not test:
        __spec__ = None

        if len(sys.argv) != 2:
            print("I need an integer to tell me which experiment to run. \n\t"
                 "Usage: python run_pf <N>")
            sys.exit(1)
    
    model_params = configs.model_params
    ukf_params = configs.ukf_params
    
    model_params, ukf_params, base_model = ex_input(model_params, ukf_params, test)
    
    print("UKF params: " + str(ukf_params))
    print("Model params: " + str(model_params))
    
    #init and run ukf
    u = ukf_ss(model_params, ukf_params, base_model)
    u.main()
        
    if ukf_params["do_pickle"]:
        #store final class instance via pickle
        f_name = ukf_params["file_name"]
        pickler(u, "results/", f_name)
    
    else:
        #alternatively save whatever necessary using some ex_save function.
        # requires, the instance and some directory + name to save to.
        f_name = ukf_params["f_name"]
        ex_save(u, "results/", ukf_params["f_name"])

    if test:
        print(f"Test successful. Deleting the saved test file : {f_name}")
        remove("results/" + ukf_params["full_file_name"])
    
if __name__ == '__main__':
    
    main(ex0_input, ex0_save)
    main(ex1_input)
    main(ex2_input)


