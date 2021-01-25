"""
main.py
@author: ksuchak1990
Python script for running experiments with the enkf.
"""

# Imports
import numpy as np
from experiment_utils import Modeller, Visualiser

np.random.seed(42)

# Functions
# def testing():
    # """
    # Testing function

    # Overall function that wraps around what we want to run at any specific
    # time.
    # """
    # with open('results/data.json') as json_file:
        # data = json.load(json_file)
    # forecasts, analyses, observations = process_repeat_results(data)
    # plot_all_results(forecasts, analyses, observations)
    # plot_with_errors(forecasts, analyses, observations)
    # run_repeat_combos(resume=True)
    # run_repeat_combos_mt(4)


# testing()
# process_batch(read_time=True)

# d = {'station': 'Grand_Central'}
# Modeller.run_repeat_combos(resume=False)
# Modeller.run_for_endtime()
# Modeller.run_experiment_1()
# Modeller.run_all(ensemble_size=10)
# Modeller.run_enkf_benchmark(ensemble_size=50, pop_size=50)
# Visualiser.quick_plot()
# Modeller.run_experiment_1_1()
Modeller.run_repeat_combos_2(resume=True)
