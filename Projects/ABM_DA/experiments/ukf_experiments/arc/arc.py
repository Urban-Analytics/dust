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
#path = os.path.join(os.path.dirname(__file__), os.pardir)
#sys.path.append(path)

sys.path.append('../../../stationsim')

from ukf2 import ukf_ss, pickler
from stationsim_model import Model

from default_ukf_configs import model_params,ukf_params
from ukf_ex0 import ex0_params, ex0_save
from ukf_ex1 import omission_params
from ukf_ex2 import aggregate_params

import numpy as np

#%%
def ex0_input(model_params,ukf_params):
    """ experiment 0 input for arc
    
    """
    
    num_age = 10 # 10 to 30 agent population by 10

    sample_rate = [1,2,5,10] # how often to assimilate with ukf
    noise = [0,0.25,0.5,1,2,5] #list of gaussian noise standard deviations
    run_id = np.arange(0,30,1)  # 30 runs

    param_list = [(x, y, z) for x in sample_rate for y in noise for z in run_id]
    
    model_params['pop_total'] = num_age
 
    #sample_rate = param_list[600][0]
    #noise = param_list[600][1]
    sample_rate = param_list[int(sys.argv[1])-1][0]
    noise = param_list[int(sys.argv[1])-1][1]
    
    ex0_params(num_age, noise, sample_rate, model_params, ukf_params)
    
    
    #ukf_params["run_id"] = param_list[600][2]
    ukf_params["run_id"] = param_list[int(sys.argv[1])-1][2]

    ukf_params["f_name"] = "config_agents_{}_rate_{}_noise_{}-{}".format(      
                            str(int(model_params['pop_total'])).zfill(3),
                            str(float(ukf_params['sample_rate'])),
                            str(float(ukf_params['noise'])),
                            str(ukf_params["run_id"]).zfill(3))
    
    ukf_params["do_pickle"] = False
    
    
def ex1_input(model_params,ukf_params):
    
    
    """input for experiment 1
    """
    
    num_age = [10,20,30]  # 10 to 30 agent population by 10
    props = [0.25,0.5,0.75,1]  # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
    run_id = np.arange(0,30,1)  # 30 runs
    
    param_list = [(x, y,z) for x in num_age for y in props for z in run_id]
    
    n =  param_list[int(sys.argv[1])-1][0]
    prop = param_list[int(sys.argv[1])-1][1]
    model_params['pop_total'] = n
    
    omission_params(n, prop, model_params, ukf_params)
    
    ukf_params["run_id"] = param_list[int(sys.argv[1])-1][2]
    ukf_params["f_name"] = "ukf_results/ukf_agents_{}_prop_{}-{}".format(      
                            str(int(model_params['pop_total'])).zfill(3),
                            str(float(ukf_params['prop'])),
                            str(ukf_params["run_id"]).zfill(3))
    
    ukf_params["do_pickle"] = True

def ex2_input(model_params,ukf_params):
    
    num_age = [10,20,30]  # 10 to 30 agent population by 10
    bin_size = [5,10,25,50]  # unitless grid square size (must be a factor of 100 and 200)
    run_id = np.arange(0,30,1)  # 30 runs
    
    param_list = [(x, y,z) for x in num_age for y in bin_size for z in run_id]
    n = param_list[int(sys.argv[1])-1][0] 
    bin_size = param_list[int(sys.argv[1])-1][1]
    model_params['pop_total'] = n
    aggregate_params(n, bin_size, model_params, ukf_params)
    
    ukf_params["run_id"] = param_list[int(sys.argv[1])-1][2]
    ukf_params["f_name"] = "ukf_results/ukf_agents_{}_bin_{}-{}".format(      
                            str(int(model_params['pop_total'])).zfill(3),
                            str(float(ukf_params['bin_size'])),
                            str(ukf_params["run_id"]).zfill(3))
    
    ukf_params["do_pickle"] = True


def main(ex_input,ex_save=None):
    __spec__ = None

    if len(sys.argv) != 2:
        print("I need an integer to tell me which experiment to run. \n\t"
             "Usage: python run_pf <N>")
        sys.exit(1)

    ex_input(model_params,ukf_params)
    #ex1_input(model_params,ukf_params)
    #ex2_input(model_params,ukf_params)

    
    print("UKF params: " + str(ukf_params))
    print("Model params: " + str(model_params))
    
    #init and run ukf
    base_model = Model(**model_params)
    u = ukf_ss(model_params, ukf_params, base_model)
    u.main()
        
    if ukf_params["do_pickle"]:
        #store final class instance via pickle
        pickler(u, "../ukf_results/", ukf_params["pickle_file_name"])
    
    else:
        ex_save(u, "../ukf_results/", ukf_params["f_name"])
    
if __name__ == '__main__':
    main(ex0_input, ex0_save)
    #main(ex1_input)
    #main(ex2_input)
        

