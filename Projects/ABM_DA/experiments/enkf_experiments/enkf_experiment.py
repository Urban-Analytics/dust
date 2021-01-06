"""
main.py
@author: ksuchak1990
Python script for running experiments with the enkf.
"""

# Imports
import numpy as np
from experiment_utils import Modeller

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
Modeller.run_all(its=1000, pop_size=25, ensemble_size=10,
                 assimilation_period=50)
# Modeller.run_repeat_combos(resume=False)
# Modeller.run_for_endtime()
# Modeller.run_experiment_1()
