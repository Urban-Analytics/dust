
"""
Run some experiments using the stationsim/particle_filter.
The script
@author: RC
"""

"""
to get this to run needs a virtual environment in arc

module load python python-libs
virtualenv mypython
source mypython/bin/activate

any additional files can be installed using pip
e.g. 
pip install imageio
pip install ffmpeg
are only two needed for animatations afaik
"""

# Need to append the main project directory (ABM_DA) and stationsim folders to the path, otherwise either
# this script will fail, or the code in the stationsim directory will fail.
import sys
#sys.path.append('../../stationsim')
#sys.path.append('../..')
from ukf import ukf_ss
from stationsim_model import Model

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

    # Lists of particles, agent numbers, and particle noise levels
    
    #num_par = list([1] + list(range(10, 50, 10)) + list(range(100, 501, 100)) + list(range(1000, 2001, 500)) + [3000, 5000, 7500, 10000])
    #num_age = np.arange(5,105,5)
    #props = np.arange(0.1,1.1,0.1)
    num_age = np.arange(5,15,5)
    props = np.arange(0.5,1.5,0.5)
    #noise = [1.0, 2.0]

    # List of all particle-agent combinations. ARC task
    # array variable loops through this list
    param_list = [(x, y) for x in num_age for y in props]

    # Use below to update param_list if some runs abort
    # If used, need to update ARC task array variable
    #
    # aborted = [2294, 2325, 2356, 2387, 2386, 2417, 2418, 2448, 2449, 2479, 2480, 2478, 2509, 2510, 2511, 2540, 2541, 2542]
    # param_list = [param_list[x-1] for x in aborted]
    
    model_params = {
			'pop_total': param_list[int(sys.argv[1]) - 1][0],

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
            'sample_rate': 1,
            "do_restrict": True, 
            "do_animate": False,
            "do_wiggle_animate": False,
            "do_density_animate":True,
            "do_pair_animate":False,
            "prop": param_list[int(sys.argv[1]) - 1][1],
            "heatmap_rate": 1,
            "bin_size":10,
            "do_batch":False,
            "do_unobserved":True,
            "number_of_runs": 20
            }
    
    ukf_params = {
            
            "a":1,
            "b":2,
            "k":0,
            "d_rate" : 10, #data assimilotion rate every "d_rate model steps recalibrate UKF positions with truth

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

    for i in range(filter_params['number_of_runs']):
    #    os.chdir("ukf_results")
        # Run the particle filter
        np.random.seed(8)
        start_time = time.time()  # Time how long the whole run take
        base_model = Model(**model_params)
        u = ukf_ss(model_params,filter_params,ukf_params,base_model)
        u.main()
        
        f_name = "ukf_agents_{}_prop_{}-{}.csv".format(      
        str(int(model_params['pop_total'])),
        str(filter_params['prop']),
        str(i))
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

        print("Run: {}, prop: {}, agents: {}, took: {}(s)".format(i, filter_params['prop'], 
              model_params['pop_total'], round(time.time() - start_time),flush=True))

    #os.chdir("..")    
    print("Finished single run")
