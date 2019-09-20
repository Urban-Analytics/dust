
"""
Run some experiments using the stationsim/particle_filter.
The script
@author: RC

run following in bash console:
    
ssh username@arc3.leeds.ac.uk
git clone https://github.com/Urban-Analytics/dust/
cd /nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments

module load python python-libs
virtualenv mypython
# if virtualenv already exists just use this line. 
# MAY WANT TO PIP THE BELOW ALREADY ANYWAY
source mypython/bin/activate

pip install imageio
pip install filterpy
pip install ffmpeg
pip install seaborn
#(and any other dependencies)
 
qsub arc_base_config.sh
"""

"""
when exporting from arc in another linux terminal use 

scp username@leeds.ac.uk:source_in_arc/* destination_in_linux/.

e.g.

from linux terminal 
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agents* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
"""


# Need to append the main project directory (ABM_DA) and stationsim folders to the path, otherwise either
# this script will fail, or the code in the stationsim directory will fail.
import sys
sys.path.append('../../stationsim')
sys.path.append('../..')
from base_config_ukf import ukf_ss
from stationsim.stationsim_model import Model
from stationsim.ukf import plots
import pickle

import os
import numpy as np

if __name__ == '__main__':
    __spec__ = None

    if len(sys.argv) != 2:
        print("I need an integer to tell me which experiment to run. \n\t"
                 "Usage: python run_pf <N>")
        sys.exit(1)

    # Lists of particles, agent numbers, and particle noise levels
    
    #num_par = list([1] + list(range(10, 50, 10)) + list(range(100, 501, 100)) + list(range(1000, 2001, 500)) + [3000, 5000, 7500, 10000])
    #num_age = np.arange(5,105,5)
    #props = np.arange(0.1,1.1,0.1)
    num_age = np.arange(10,70,20) # 5 to 50 by 5
    rates = [1,2,5,10,20,50,100] #.2 to 1 by .2
    noise = [0,0.25,0.5,1,2,5,10,25,50,100]
    run_id = np.arange(0,20,1) #20 runs
    #noise = [1.0, 2.0]

    # List of all particle-agent combinations. ARC task
    # array variable loops through this list
    param_list = [(x, y,z,a) for x in num_age for y in rates for z in noise for a in run_id]

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
            'sample_rate': param_list[int(sys.argv[1])-1][1],
            "do_restrict": True, 
            "do_animate": False,
            "prop": 1,
            "run_id":param_list[int(sys.argv[1])-1][3],
            "heatmap_rate": 1,
            "bin_size":25,
            "bring_noise":True,
            "noise":param_list[int(sys.argv[1])-1][2],
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
    base_model = Model(**model_params)
    u = ukf_ss(model_params,filter_params,ukf_params,base_model)
    u.main()
    true,obs,preds,histories= u.data_parser(True)
    plts=plots(u)
    errors = {}
    errors["obs"] = plts.L2s(true,obs)
    errors["preds"] = plts.L2s(true[1:,:],preds)
    errors["ukf"] = plts.L2s(true[1:,:],histories)
    
    means = []
    for key in errors.keys():
        means.append(np.nanmean(errors[key][0]))
    
    means = np.array(means)
    n = model_params["pop_total"]
    r = filter_params["sample_rate"]
    var = filter_params["noise"]
    run_id = filter_params["run_id"]
    
    f_name = "ukf_results/agents_{}_rate_{}_noise_{}_base_config_errors_{}".format(str(n),str(r),str(var), str(run_id))
    f = open(f_name,"wb")
    pickle.dump(means,f)
    f.close()
