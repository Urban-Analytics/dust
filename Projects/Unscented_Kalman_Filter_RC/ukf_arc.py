
# -*- coding: utf-8 -*-
"""
Run some experiments using the stationsim/particle_filter.
The script
@author: medkmin
@author: nickmalleson
"""

# Need to append the main project directory (ABM_DA) and stationsim folders to the path, otherwise either
# this script will fail, or the code in the stationsim directory will fail.
import sys
sys.path.append('../../stationsim')
sys.path.append('../..')
from stationsim.particle_filter import ParticleFilter
from stationsim.stationsim_model import Model

import os
import time
import warnings
import multiprocessing

if __name__ == '__main__':
    __spec__ = None

    if len(sys.argv) != 2:
        print("I need an integer to tell me which experiment to run. \n\t"
                 "Usage: python run_pf <N>")
        sys.exit(1)

    # Lists of particles, agent numbers, and particle noise levels
    num_par = list([1] + list(range(10, 50, 10)) + list(range(100, 501, 100)) + list(range(1000, 2001, 500)) + [3000, 5000, 7500, 10000])
    num_age = [2, 5, 10, 15, 20, 30, 40, 50]
    noise = [1.0, 2.0]

    # List of all particle-agent combinations. ARC task
    # array variable loops through this list
    param_list = [(x, y, z) for x in num_par for y in num_age for z in noise]

    # Use below to update param_list if some runs abort
    # If used, need to update ARC task array variable
    #
    # aborted = [2294, 2325, 2356, 2387, 2386, 2417, 2418, 2448, 2449, 2479, 2480, 2478, 2509, 2510, 2511, 2540, 2541, 2542]
    # param_list = [param_list[x-1] for x in aborted]

    model_params = {
        'width': 200,
        'height': 100,
        'pop_total': param_list[int(sys.argv[1]) - 1][1],  # agents read from ARC task array variable
        'entrances': 3,
        'entrance_space': 2,
        'entrance_speed': .1,
        'exits': 2,
        'exit_space': 1,
        'speed_min': .1,
        'speed_desire_mean': 1,
        'speed_desire_std': 1,
        'separation': 2,
        'batch_iterations': 4000,  # Only relevant in batch() mode
        'do_save': True,  # Saves output data (only relevant in batch() mode)
        'do_ani': False,  # Animates the model (only relevant in batch() mode)
        'do_history': False,
        'do_print': False,
    }
    # Model(model_params).batch() # Runs the model as normal (one run)

    filter_params = {
        'number_of_particles': param_list[int(sys.argv[1]) - 1][0],  # particles read from ARC task array variable
        'number_of_runs': 20,  # Number of times to run each particle filter configuration
        'resample_window': 100,
        'multi_step': True,  # Whether to predict() repeatedly until the sampling window is reached
        'particle_std': param_list[int(sys.argv[1]) - 1][2], # Particle noise read from task array variable
        'model_std': 1.0,  # was 2 or 10
        'agents_to_visualise': 10,
        'do_save': True,
        'plot_save': False,
        'do_ani': False,

    }

    # Open a file to write the results to
    # The results directory should be in the parent directory of this script
    # results_dir = os.path.join(sys.path[0], "..", "results")  # sys.path[0] gets the location of this file
    # The results directory should be in the same location as the caller of this script
    results_dir = os.path.join("./results")
    if not os.path.exists(results_dir):
        raise FileNotFoundError("Results directory ('{}') not found. Are you ".format(results_dir))
    outfile = os.path.join(results_dir, "pf_particles_{}_agents_{}_noise_{}-{}.csv".format(
        str(int(filter_params['number_of_particles'])),
        str(int(model_params['pop_total'])),
        str(filter_params['particle_std']),
        str(int(time.time()))
    ))
    with open(outfile, 'w') as f:
        # Write the parameters first
        f.write("PF params: " + str(filter_params) + "\n")
        f.write("Model params: " + str(model_params) + "\n")
        # Now write the csv headers
        f.write(
            "Min_Mean_errors,Max_Mean_errors,Average_mean_errors,Min_Absolute_errors,Max_Absolute_errors,Average_Absolute_errors,Min_variances,Max_variances,Average_variances,Before_resample?\n")


    print("PF params: " + str(filter_params))
    print("Model params: " + str(model_params))
    print("Saving files to: {}".format(outfile))

    for i in range(filter_params['number_of_runs']):

        # Run the particle filter

        start_time = time.time()  # Time how long the whole run take
        pf = ParticleFilter(Model, model_params, filter_params, numcores = int(multiprocessing.cpu_count()))
        result = pf.step()


        with open(outfile, 'a') as f:
            if result == None: # If no results then don't write anything
                warnings.warn("Result from the particle filter is 'none' for some reason. This sometimes happens when there is only 1 agent in the model.")
            else:
                # Two sets of errors are created, those before resampling, and those after. Results is a list with two tuples.
                # First tuple has eerrors before resampling, second has errors afterwards.
                for before in [0, 1]:
                    f.write(str(result[before])[1:-1].replace(" ", "") + "," + str(before) + "\n")  # (slice to get rid of the brackets aruond the tuple)

        print("Run: {}, particles: {}, agents: {}, took: {}(s), result: {}".format(
            i, filter_params['number_of_particles'], model_params['pop_total'], round(time.time() - start_time),
            result), flush=True)

    print("Finished single run")