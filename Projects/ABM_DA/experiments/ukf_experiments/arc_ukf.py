
"""
Run some experiments using the stationsim/particle_filter.
The script
@author: RC

run following in bash console:
    
ssh username@arc3.leeds.ac.uk
git clone https://github.com/Urban-Analytics/dust/
cd dust/Projects/ABM_DA/experiments/ukf_experiments

module load python python-libs
virtualenv mypython
source mypython/bin/activate

pip install imageio
pip install filterpy
pip install ffmpeg
pip install seaborn
#(and any other dependencies)
 
qsub arc_ukf.sh
"""

"""
when exporting from arc in another linux terminal use 

scp username@leeds.ac.uk:source_in_arc/* destination_in_linux/.

e.g.

from linux terminal 
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/ukf* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
"""


# Need to append the main project directory (ABM_DA) and stationsim folders to the path, otherwise either
# this script will fail, or the code in the stationsim directory will fail.
import sys
sys.path.append('../../stationsim')
sys.path.append('../..')
from stationsim.ukf import ukf_ss
from stationsim.stationsim_model import Model

import os
import time
import warnings
import numpy as np
import pickle

if __name__ == '__main__':
    __spec__ = None

    if len(sys.argv) != 2:
        print("I need an integer to tell me which experiment to run. \n\t"
                 "Usage: python run_pf <N>")
        sys.exit(1)

    # Lists of agent numbers, and noise levels and run ids
    
    num_age = [10,20,30]# 10 to 30 by 10
    props = [0.25,0.5,0.75,1] #.25 to 1 by .25
    run_id = np.arange(0,30,1) #30 runs

    # List of all particle-agent combinations. ARC task
    # array variable loops through this list
    
    param_list = [(x, y,z) for x in num_age for y in props for z in run_id]

    # Use below to update param_list if some runs abort
    # If used, need to update ARC task array variable
    #
    # aborted = [2294, 2325, 2356, 2387, 2386, 2417, 2418, 2448, 2449, 2479, 2480, 2478, 2509, 2510, 2511, 2540, 2541, 2542]
    # param_list = [param_list[x-1] for x in aborted]
    
    model_params = {
			'pop_total': param_list[int(sys.argv[1])-1][0],

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

    filter_params = {      
           
            "Sensor_Noise":  1, 
            "Process_Noise": 1, 
            'sample_rate': 5,
            "do_restrict": True, 
            "do_animate": False,
            "prop": param_list[int(sys.argv[1])-1][1],
            "run_id":param_list[int(sys.argv[1])-1][2],
            "heatmap_rate": 1,
            "bin_size":25,
            "bring_noise":True,
            "noise":0.5,
            "do_batch":False,
            }
    
    ukf_params = {
        
            "a":1,
            "b":2,
            "k":0,
            }

    # Open a file to write the results to
    # The results directory should be in the parent directory of this script
    # results_dir = os.path.join(sys.path[0], "..", "results")  # sys.path[0] gets the location of this file
    # The results directory should be in the same location as the caller of this script
    results_dir = os.path.join("ukf_results")
    if not os.path.exists(results_dir):
        raise FileNotFoundError("Results directory ('{}') not found. Are you ".format(results_dir))
    #outfile = os.path.join(results_dir, "ukf_agents_{}_prop_{}-{}.csv".format(      
    #    str(int(model_params['pop_total'])),
    #    str(filter_params['prop']),
    #    str(int(time.time()))
    #))
    #with open(outfile, 'w') as f:
    #    # Write the parameters first
    #    f.write("PF params: " + str(filter_params) + "\n")
    #    f.write("Model params: " + str(model_params) + "\n")
    #    # Now write the csv headers
    #    f.write(
    #        "Min_Mean_errors,Max_Mean_errors,Average_mean_errors,Min_Absolute_errors,Max_Absolute_errors,Average_Absolute_errors,Min_variances,Max_variances,Average_variances,Before_resample?\n")


    print("UKF params: " + str(filter_params))
    print("Model params: " + str(model_params))
    
    #print("Saving files to: {}".format(outfile))

    # Run the particle filter
    
    #init and run ukf
    start_time = time.time()  # Time how long the whole run take
    base_model = Model(**model_params)
    u = ukf_ss(model_params,filter_params,ukf_params,base_model)
    u.main()
    
    #store final class instance via pickle
    f_name = "ukf_results/ukf_agents_{}_prop_{}-{}".format(      
    str(int(model_params['pop_total'])),
    str(filter_params['prop']),
    str(filter_params["run_id"]).zfill(3))
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

    print("Run: {}, prop: {}, agents: {}, took: {}(s)".format(str(filter_params["run_id"]).zfill(4), filter_params['prop'], 
          model_params['pop_total'], round(time.time() - start_time),flush=True))
    
    print("Finished single run")