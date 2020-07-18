# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 09:28:20 2020

@author: vijay
"""
from particle_filter_gcs_orig import ParticleFilter
#from stationsim_gcs_model import Model
from stationsim_gcs_model_orig import Model
import os
import glob
import time
import warnings
import multiprocessing
import datetime

t0=time.time()


model_params = {'pop_total': 274, 'batch_iterations': 3100, 'step_limit': 3100, 'birth_rate': 25./15, 'do_history': False, 'do_print': True
                , 'station': 'Grand_Central'}
filter_params = {'agents_to_visualise': 100, 'number_of_runs': 1, 'multi_step': False, 'particle_std': 1.0, 'model_std': 1.0, 'do_save': True, 'plot_save': False,
                 'do_ani': False, 'show_ani': False, 'do_external_data': True, 'resample_window': 1,
                 'number_of_particles': 5,
                 'do_resample': True, # True for experiments with D.A.
                 'external_info': ['gcs_final_real_data/', True, True]}  # [Real data dir, Use external velocit?, Use external gate_out?]

pf = ParticleFilter(Model, model_params, filter_params, numcores = int(multiprocessing.cpu_count()))
result = pf.step()