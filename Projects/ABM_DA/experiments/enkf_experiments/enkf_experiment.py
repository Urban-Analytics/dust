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
# Modeller.run_all(its=3600, pop_size=10)
Modeller.run_repeat_combos(resume=False)
