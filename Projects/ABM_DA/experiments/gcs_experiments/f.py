# This is a workaround to allow multiprocessing.Pool to work in the pf_experiments_plots notebook.
# The function called by pool.map ('count_wiggles') needs to be defined in this separate file and imported.
# https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397
import sys
sys.path.append('../../stationsim')
from stationsim_gcs_model import Model

def count_wiggles(num_agents, mp): 
    """Run a model and return the num. agents and num. wiggles. mp is the model parameters (dictionary)"""
    mp['pop_total'] = num_agents # Set the number of agents for this model
    mp['do_print']  = False # Don't print anything
    m = Model(**mp)
    for _ in range(m.step_limit):
        m.step()
    return ( num_agents, len(m.history_wiggle_locs) )