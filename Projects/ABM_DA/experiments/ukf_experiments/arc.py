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
ssh username@arc3.leeds.ac.uk
#move to nobackup storage
cd /nobackup/username
#clone in git repository
git clone https://github.com/Urban-Analytics/dust/
#move to experiments folder
cd /dust/Projects/ABM_DA/experiments/ukf_experiments

#create python 3 virtual environment
module load python python-libs
virtualenv mypython
source mypython/bin/activate


#install in dependencies.
# for filtering
pip install filterpy
# for plotting
pip install seaborn
# for animations
pip install imageio
# 
# ffmpeg-imageio not on pip. either install an older imageio version <=2.4.1 
# or find another way e.g.
# conda install -c conda-forge imageio-ffmpeg ""
# either way. this isnt necessary for arc only. 
#
qsub arc_ukf.sh

when exporting from arc in another linux terminal use 

scp username@leeds.ac.uk:source_in_arc/* destination_in_linux/.

e.g.

from linux terminal 
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/ukf* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
"""

import os
import sys
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)


sys.path.append('../..')

from stationsim.ukf2 import ukf_ss
from stationsim.stationsim_model import Model
from ukf_ex1 import omission_params
from ukf_ex2 import aggregate_params
import os
import datetime
import warnings
import numpy as np
import pickle

#%%

def ex1_input(num_age,props,run_id):

    param_list = [(x, y,z) for x in num_age for y in props for z in run_id]
    model_params['pop_total'] = param_list[int(sys.argv[1])-1][0]
    omission_params(model_params,ukf_params, param_list[int(sys.argv[1])-1][1])
    ukf_params["run_id"] = param_list[int(sys.argv[1])-1][2]
    ukf_params["f_name"] = "ukf_results/ukf_agents_{}_prop_{}-{}".format(      
                            str(int(model_params['pop_total'])).zfill(3),
                            str(float(ukf_params['prop'])),
                            str(ukf_params["run_id"]).zfill(3))
    
def ex2_input(num_age,bin_size,run_id):

    param_list = [(x, y,z) for x in num_age for y in bin_size for z in run_id]
    model_params['pop_total'] = param_list[int(sys.argv[1])-1][0]
    aggregate_params(model_params, ukf_params, param_list[int(sys.argv[1])-1][1])
    ukf_params["run_id"] = param_list[int(sys.argv[1])-1][2]
    ukf_params["f_name"] = "ukf_results/agg_ukf_agents_{}_bin_{}-{}".format(      
                                str(int(model_params['pop_total'])).zfill(3),
                                str(ukf_params['bin_size']).zfill(3),
                                str(ukf_params["run_id"]).zfill(3))

if __name__ == '__main__':
    __spec__ = None

    if len(sys.argv) != 2:
        print("I need an integer to tell me which experiment to run. \n\t"
                 "Usage: python run_pf <N>")
        sys.exit(1)

    model_params = {

			'width': 200,
			'height': 100,

			'gates_in': 3,
			'gates_out': 2,
			'gates_space': 1,
			'gates_speed': 1,

			'speed_min': .2,
			'speed_mean': 1,
			'speed_std': 1,
			'speed_steps': 3,

			'separation': 5,
			'max_wiggle': 1,

			'step_limit': 3600,

			'do_history': True,
			'do_print': True,
		}

    ukf_params = {      
            'sample_rate': 5,
            "do_restrict": True, 
            "do_animate": False,         
            
            "bring_noise":True,
            "noise":0.5,
            "do_batch":False,

            "a":1,
            "b":2,
            "k":0,
            }

    num_age = [10,20,30]  # 10 to 30 agent population by 10
    props = [0.25,0.5,0.75,1]  # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
    #bin_size = [5,10,25,50]  # unitless grid square size (must be a factor of 100 and 200)
    run_id = np.arange(0,30,1)  # 30 runs
    ex1_input(num_age, props, run_id,)
    #ex2_input(num_age,bin_size,run_id)

    
    print("UKF params: " + str(ukf_params))
    print("Model params: " + str(model_params))
    
    #print("Saving files to: {}".format(outfile))

    # Run the particle filter
    
    #init and run ukf
    time1 = datetime.datetime.now()  # Time how long the whole run take
    base_model = Model(**model_params)
    u = ukf_ss(model_params, ukf_params, base_model)
    u.main()
    
    #store final class instance via pickle
    f_name = ukf_params["f_name"]
    f = open(f_name,"wb")
    pickle.dump(u,f)
    f.close()


        #with open(outfile, 'a') as f:
        #    if result == None: # If no results then don't write anything
        #        warnings.warn("Result from the particle filter is 'none' for some reason. This sometimes happens when there is only 1 agent in the model.")
        #    else:
        #        # Two sets of errors are created, those before resampling, and those after. Results is a list with two tuples.
        #        # First tuple has eerrors before resampling, second has errors afterwards.
        #        for before in [0, 1]:
        #            f.write(str(result[before])[1:-1].replace(" ", "") + "," + str(before) + "\n")  # (slice to get rid of the brackets aruond the tuple)

    print("Run: {}, prop: {}, agents: {}, took: {}(s)".format(str(ukf_params["run_id"]).zfill(4), ukf_params['prop'], 
          model_params['pop_total'], round(time1 - datetime.datetime.now()),flush=True))
    
    print("Finished single run")


