"""
run.py
author: ksuchak1990
A collection of functions for running the enkf.
"""

# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import queue
import threading
import sys
from time import sleep

sys.path.append('../../stationsim')
from stationsim_gcs_model import Model
from ensemble_kalman_filter import EnsembleKalmanFilter


# Functions
def run_repeat_combos_mt(num_worker_threads=2):
    print('getting started with {0} threads'.format(num_worker_threads))

    def do_work():
        i = 0
        tn = threading.current_thread().getName()
        print('starting {0}'.format(tn))
        while True:
            item = q.get()
            if item is None:
                break

            if i % 40 == 25:
                print('Taking a short break.\n\n\n')
                sleep(30)
            if i % 200 == 123:
                print('Taking a long break.\n\n\n')
                sleep(120)

            ts = ''
            for k, v in item.items():
                ts = ts + '{0}: {1} '.format(k, v)

            print('{0} is running {1}'.format(tn, ts))

            run_repeat(**item)

            i += 1
            q.task_done()

    ap = [2, 5, 10, 20, 50]
    es = [2, 5, 10, 20, 50, 100]
    pop = [1+(5 * i) for i in range(0, 11)]
    sigma = [0.5 * i for i in range(1, 6)]

    combos = list()

    for a in ap:
        for e in es:
            for p in pop:
                for s in sigma:
                    t = {'a': a, 'e': e, 'p': p,
                         's': s, 'N': 10, 'write_json': True}
                    # t = (a, e, p, s)
                    combos.append(t)

    # combos.reverse()

    q = queue.Queue()
    for c in combos:
        q.put(c)

    print('{} jobs to complete'.format(len(combos)))
    threads = list()

    for _ in range(num_worker_threads):
        t = threading.Thread(target=do_work)
        threads.append(t)
        t.start()

    q.join()
    for _ in range(num_worker_threads):
        q.put(None)
    for t in threads:
        t.join()

def run_enkf(model_params, filter_params):
    """
    Run Ensemble Kalman Filter on model using the generated noisy data.
    """
    # Initialise filter with StationSim and params
    enkf = EnsembleKalmanFilter(Model, filter_params, model_params)

    num_steps = filter_params['max_iterations']

    for i in range(num_steps):
        if i % 25 == 0:
            print('step {0}'.format(i))
        enkf.step()

        agents_active = [m.pop_active for m in enkf.models]
        agents_finished = [a==0 for a in agents_active]
        a = all(agents_finished)
        b = i > num_steps//10
        c = enkf.base_model.pop_active==0
        if a and b and c:
            print('Filter finished after {0} iterations'.format(i))
            break

    return enkf


def run_all(pop_size=20, its=300, assimilation_period=50, ensemble_size=10):
    """
    Overall function to run everything.
    """
    # Set up params
    # model_params = {'pop_total': 5,
                    # 'station': 'Grand_Central',
                    # 'do_print': True}
    model_params = {'width': 200,
                    'height': 100,
                    'pop_total': pop_size,
                    'gates_in': 3,
                    'gates_space': 2,
                    'gates_speed': 4,
                    'gates_out': 2,
                    'speed_min': .1,
                    'speed_mean': 1,
                    'speed_std': 1,
                    'speed_steps': 3,
                    'separation': 4,
                    'max_wiggle': 1,
                    'step_limit': its,
                    'do_history': True,
                    'do_print': False}

    OBS_NOISE_STD = 1
    vec_length = 2 * model_params['pop_total']

    filter_params = {'max_iterations': its,
                     'assimilation_period': assimilation_period,
                     'ensemble_size': ensemble_size,
                     'state_vector_length': vec_length,
                     'data_vector_length': vec_length,
                     'H': np.identity(vec_length),
                     'R_vector': OBS_NOISE_STD * np.ones(vec_length),
                     'keep_results': True,
                     'run_vanilla': True,
                     'vis': False}

    # Run enkf and process results
    enkf = run_enkf(model_params, filter_params)

    print('base', enkf.base_model.step_id, len(enkf.base_model.history_state))
    for i, m in enumerate(enkf.models):
        print('model {0}'.format(i), m.step_id, len(m.history_state))

    print(enkf.base_model.time_save)

    enkf.process_results()

    # r = pd.DataFrame(enkf.rmse)
    # plt.figure()
    # plt.plot(r['time'], r['obs'], label='observations')
    # plt.plot(r['time'], r['analysis'], label='filter')
    # plt.plot(r['time'], r['vanilla'], label='vanilla')
    # plt.legend()
    # plt.show()

def run_repeat(a=50, e=10, p=20, s=1, N=10, write_json=False):
    """
    Repeatedly run an enkf realisation of stationsim.

    Run a realisation of stationsim with the enkf repeatedly.
    Produces RMSE values for forecast, analysis and observations at each
    assimilation step.

    Parameters
    ----------
    N : int
        The number of times we want to run the ABM-DA.
    """
    model_params = {'width': 200,
                    'height': 100,
                    'pop_total': p,
                    'gates_in': 3,
                    'gates_space': 2,
                    'gates_speed': 4,
                    'gates_out': 2,
                    'speed_min': .1,
                    'speed_mean': 1,
                    'speed_std': 1,
                    'speed_steps': 3,
                    'separation': 4,
                    'max_wiggle': 1,
                    'step_limit': 500,
                    'do_history': True,
                    'do_print': False}

    OBS_NOISE_STD = s
    vec_length = 2 * model_params['pop_total']

    filter_params = {'max_iterations': model_params['step_limit'],
                     'assimilation_period': a,
                     'ensemble_size': e,
                     'state_vector_length': vec_length,
                     'data_vector_length': vec_length,
                     'H': np.identity(vec_length),
                     'R_vector': OBS_NOISE_STD * np.ones(vec_length),
                     'keep_results': True,
                     'vis': False}

    errors = list()
    for i in range(N):
        print('Running iteration {0}'.format(i+1))
        enkf = run_enkf(model_params, filter_params)
        errors.append(enkf.rmse)

    if write_json:
        fname = 'data_{0}__{1}__{2}__{3}'.format(a, e, p, s).replace('.', '_')
        with open('results/repeats/{0}.json'.format(fname), 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=4)
    return errors


def run_combos():
    """
    Run the ensemble kalman filter for different combinations of:
        - assimilation period
        - ensemble size
    """
    # Parameter combos
    assimilation_periods = [2, 5, 10, 20, 50, 100]
    ensemble_sizes = [2, 5, 10, 20]

    for a in assimilation_periods:
        for e in ensemble_sizes:
            combo = (20, 300, a, e)
            run_all(*combo)

def run_repeat_combos(resume=True):
    """
    Repeatedly run model + enkf for a range of different parameter
    combinations.

    Run the model + enkf 10 times for each combination of parameter values.
    If some combinations have already been run then the option to resume can be
    used.
    Write outputs to json every 5 combinations.

    Parameters
    ----------
    resume : boolean
        Boolean to choose whether to resume from previous point in the list of
        combinations.
    """
    if resume:
        with open('results/combos.json') as json_file:
            combos = json.load(json_file)
    else:
        ap = [2, 5, 10, 20, 50]
        es = [2, 5, 10, 20, 50, 100]
        pop = [1+(5 * i) for i in range(0, 11)]
        sigma = [0.5 * i for i in range(1, 6)]

        combos = list()

        for a in ap:
            for e in es:
                for p in pop:
                    for s in sigma:
                        t = (a, e, p, s)
                        combos.append(t)

    i = 0
    combos.reverse()
    while combos:
        c = combos.pop()
        print('running for {0}'.format(str(c)))
        run_repeat(*c, N=10, write_json=True)
        if i % 5 == 0:
            with open('results/combos.json', 'w', encoding='utf-8') as f:
                json.dump(combos, f, ensure_ascii=False, indent=4)
        if i % 40 == 25:
            print('Taking a short break.\n\n\n')
            sleep(30)
        if i % 200 == 123:
            print('Taking a long break.\n\n\n')
            sleep(300)
        if i == 1000:
            break
        i += 1

