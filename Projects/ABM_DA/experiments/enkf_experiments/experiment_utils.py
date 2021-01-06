"""
experiment_utils.py
author: ksuchak1990
A collection of classes and functions for running the enkf.
"""

# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
from pathlib import Path
import queue
import seaborn as sns
import threading
import sys
from time import sleep

sys.path.append('../../stationsim')
from stationsim_gcs_model import Model
from ensemble_kalman_filter import EnsembleKalmanFilter
from ensemble_kalman_filter import EnsembleKalmanFilterType


# Classes
class Modeller():
    def __init__(self):
        pass

    @classmethod
    def run_all(cls, pop_size=20, its=300, assimilation_period=50,
                ensemble_size=10, mode=EnsembleKalmanFilterType.DUAL_EXIT):
        """
        Overall function to run everything.
        """
        # Set up params
        model_params = {'pop_total': pop_size,
                        'station': 'Grand_Central',
                        'do_print': True}
        # model_params = {'width': 200,
                        # 'height': 100,
                        # 'pop_total': pop_size,
                        # 'gates_in': 3,
                        # 'gates_space': 2,
                        # 'gates_speed': 4,
                        # 'gates_out': 2,
                        # 'speed_min': .1,
                        # 'speed_mean': 1,
                        # 'speed_std': 1,
                        # 'speed_steps': 3,
                        # 'separation': 4,
                        # 'max_wiggle': 1,
                        # 'step_limit': its,
                        # 'do_history': True,
                        # 'do_print': False}

        # Set up filter parameters
        OBS_NOISE_STD = 1
        observation_operator = cls.__make_observation_operator(pop_size, mode)
        state_vec_length = cls.__make_state_vector_length(pop_size, mode)
        data_mode = EnsembleKalmanFilterType.STATE
        data_vec_length = cls.__make_state_vector_length(pop_size, data_mode)

        filter_params = {'max_iterations': its,
                         'assimilation_period': assimilation_period,
                         'ensemble_size': ensemble_size,
                         'population_size': pop_size,
                         'state_vector_length': state_vec_length,
                         'data_vector_length': data_vec_length,
                         'mode': mode,
                         'H': observation_operator,
                         'R_vector': OBS_NOISE_STD * np.ones(data_vec_length),
                         'keep_results': True,
                         'run_vanilla': False,
                         'vis': False}

        # Run enkf and process results
        enkf = cls.run_enkf(model_params, filter_params)
        for agent in enkf.base_model.agents:
            print(agent.gate_out)

        # Plotting
        Visualiser.plot_error_timeseries(enkf, model_params,
                                         filter_params, True)
        Visualiser.plot_forecast_error_timeseries(enkf, model_params,
                                                  filter_params, True)
        Visualiser.plot_exits(enkf)

        # Plotting to look at accuracy of exit estimations
        if mode == EnsembleKalmanFilterType.DUAL_EXIT:
            Visualiser.plot_exit_accuracy(enkf, model_params)

    @classmethod
    def run_combos(cls):
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
                cls.run_all(*combo)

    @staticmethod
    def run_enkf(model_params, filter_params):
        """
        Run Ensemble Kalman Filter on model using the generated noisy data.
        """
        # Initialise filter with StationSim and params
        enkf = EnsembleKalmanFilter(Model, filter_params, model_params)

        num_steps = filter_params['max_iterations']

        for i in range(num_steps):
            # if i % 25 == 0:
                # print('step {0}'.format(i))
                # # print(enkf.models[0].get_state('loc_exit'))
            print('step {0}'.format(i))
            enkf.step()
        return enkf

    @classmethod
    def run_repeat(cls, a=50, e=10, p=20, s=1, N=10,
                   write_json=False,
                   mode=EnsembleKalmanFilterType.STATE):
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
        model_params = {'pop_total': p,
                        'station': 'Grand_Central',
                        'do_print': False}
        # model_params = {'width': 200,
                        # 'height': 100,
                        # 'pop_total': p,
                        # 'gates_in': 3,
                        # 'gates_space': 2,
                        # 'gates_speed': 4,
                        # 'gates_out': 2,
                        # 'speed_min': .1,
                        # 'speed_mean': 1,
                        # 'speed_std': 1,
                        # 'speed_steps': 3,
                        # 'separation': 4,
                        # 'max_wiggle': 1,
                        # 'step_limit': 500,
                        # 'do_history': True,
                        # 'do_print': False}

        OBS_NOISE_STD = s
        vec_length = 2 * model_params['pop_total']
        observation_operator = cls.__make_observation_operator(p, mode)

        filter_params = {'max_iterations': 3600,
                         'assimilation_period': a,
                         'ensemble_size': e,
                         'population_size': model_params['pop_total'],
                         'state_vector_length': vec_length,
                         'data_vector_length': vec_length,
                         'H': observation_operator,
                         'R_vector': OBS_NOISE_STD * np.ones(vec_length),
                         'keep_results': True,
                         'vis': False}

        errors = list()
        forecast_errors = list()
        for i in range(N):
            print('Running iteration {0}'.format(i+1))
            enkf = cls.run_enkf(model_params, filter_params)
            errors.append(enkf.rmse)
            forecast_errors.append(enkf.forecast_error)

        if write_json:
            dir_name = f'a{a}__e{e}__p{p}__s{s}'.replace('.', '_')
            full_dir = f'results/repeats/{dir_name}/'
            Path(full_dir).mkdir(parents=True, exist_ok=True)
            fname1 = f'{full_dir}/errors.json'
            fname2 = f'{full_dir}/forecast_errors.json'

            with open(fname1, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)

            with open(fname2, 'w', encoding='utf-8') as f:
                json.dump(forecast_errors, f, ensure_ascii=False, indent=2)

        return errors, forecast_errors

    @classmethod
    def run_repeat_combos(cls, resume=True):
        """
        Repeatedly run model + enkf for a range of different parameter
        combinations.

        Run the model + enkf 10 times for each combination of parameter values.
        If some combinations have already been run then the option to resume
        can be used.
        Write outputs to json every 5 combinations.

        Parameters
        ----------
        resume : boolean
            Boolean to choose whether to resume from previous point in the list
            of combinations.
        """
        if resume:
            with open('results/combos.json') as json_file:
                combos = json.load(json_file)
        else:
            # Current
            ap = [2, 5, 10, 20, 50]
            es = [2, 5, 10, 20, 50]
            pop = [2, 5, 10, 20, 50]
            sigma = [0.5 * i for i in range(1, 6)]

            # Old
            # ap = [2, 5, 10, 20, 50]
            # es = [2, 5, 10, 20, 50, 100]
            # pop = [1+(5 * i) for i in range(0, 11)]
            # sigma = [0.5 * i for i in range(1, 6)]

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
            cls.run_repeat(*c, N=10, write_json=True)
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

    # def run_repeat_combos_mt(num_worker_threads=2):
        # print('getting started with {0} threads'.format(num_worker_threads))

        # def do_work():
            # i = 0
            # tn = threading.current_thread().getName()
            # print('starting {0}'.format(tn))
            # while True:
                # item = q.get()
                # if item is None:
                    # break

                # if i % 40 == 25:
                    # print('Taking a short break.\n\n\n')
                    # sleep(30)
                # if i % 200 == 123:
                    # print('Taking a long break.\n\n\n')
                    # sleep(120)

                # ts = ''
                # for k, v in item.items():
                    # ts = ts + '{0}: {1} '.format(k, v)

                # print('{0} is running {1}'.format(tn, ts))

                # self.run_repeat(**item)

                # i += 1
                # q.task_done()

        # ap = [2, 5, 10, 20, 50]
        # es = [2, 5, 10, 20, 50, 100]
        # pop = [1+(5 * i) for i in range(0, 11)]
        # sigma = [0.5 * i for i in range(1, 6)]

        # combos = list()

        # for a in ap:
            # for e in es:
                # for p in pop:
                    # for s in sigma:
                        # t = {'a': a, 'e': e, 'p': p,
                             # 's': s, 'N': 10, 'write_json': True}
                        # # t = (a, e, p, s)
                        # combos.append(t)

        # # combos.reverse()

        # q = queue.Queue()
        # for c in combos:
            # q.put(c)

        # print('{} jobs to complete'.format(len(combos)))
        # threads = list()

        # for _ in range(num_worker_threads):
            # t = threading.Thread(target=do_work)
            # threads.append(t)
            # t.start()

        # q.join()
        # for _ in range(num_worker_threads):
            # q.put(None)
        # for t in threads:
            # t.join()

    @staticmethod
    def run_for_endtime(N=250):
        """
        Run the model a number of times to see how the number of iterations
        taken for all agents to reach their assigned exit varies with
        population size.

        Run stationsim_gcs for a range of population sizes. 
        For each population size, run the model N times, collecting the time at
        the model finds that all of its agents have reached their assigned
        exits.
        For each population size, plot a histogram of the distribution of end
        times to inspect the variation.

        Parameters
        ----------
        N : int
            The number of times for which the model is run with each population
            size.
        """
        # Current
        pop = [2, 5, 10, 20, 50, 100]

        endtimes = dict()

        i = 0
        for p in pop:
            e = list()
            for _ in range(N):
                if i % 200 == 123:
                    print('taking a quick break')
                    sleep(30)

                model_params = {'pop_total': p,
                                'station': 'Grand_Central',
                                'do_print': False}
                m = Model(unique_id=i, **model_params)
                for t in range(4000):
                    m.step()
                e.append(m.max_time)
                i += 1
            endtimes[p] = e

        Visualiser.plot_endtimes(endtimes)

    @classmethod
    def __make_observation_operator(cls, population_size, mode):
        if mode == EnsembleKalmanFilterType.STATE:
            return np.identity(2 * population_size)
        elif mode == EnsembleKalmanFilterType.DUAL_EXIT:
            return cls.__make_exit_observation_operator(population_size)
        else:
            raise ValueError(f'Unexpected filter mode: {mode}')

    @staticmethod
    def __make_exit_observation_operator(population_size):
        a = np.identity(2 * population_size)
        b = np.zeros(shape=(2 * population_size, population_size))
        return np.hstack((a, b))

    @staticmethod
    def __make_state_vector_length(population_size, mode):
        if mode == EnsembleKalmanFilterType.STATE:
            return 2 * population_size
        elif mode == EnsembleKalmanFilterType.DUAL_EXIT:
            return 3 * population_size
        else:
            raise ValueError(f'Unexpected filter mode: {mode}')

    @classmethod
    def run_experiment_1(cls):
        # cls.run_experiment_1_1()
        cls.run_experiment_1_2()

    @classmethod
    def run_experiment_1_1(cls):
        # Run model alone for multiple population sizes
        # Show that the number of collisions increases with the population size
        # Number of collisions should be scaled by the number of timesteps
        pop_sizes = range(2, 51, 2)
        N_repeats = 25
        model_results = list()
        for pop_size in pop_sizes:
            print(f'Running for {pop_size} agents')
            for _ in range(N_repeats):
                collisions = cls.__run_model(pop_size)
                if collisions is not None:
                    model_results.append(collisions)

        model_results = pd.DataFrame(model_results)
        model_results.to_csv('./results/data/model_collisions.csv')
        plt.figure()
        sns.relplot(x='population_size', y='collisions',
                    kind='line', data=model_results)
        plt.savefig('./results/figures/model_collisions.pdf')

    @staticmethod
    def run_experiment_1_2():
        # Run the EnKF+model for multiple population sizes (multiple times)
        pop_sizes = [2, 5, 10, 20, 50, 100]
        N = 50
        its = 3600
        assimilation_period = 25
        ensemble_size = 10
        mode = EnsembleKalmanFilterType.STATE

        # Plot facet grid by population size
        # Individual plots are sns.relplot, x=time, y=error
        # Two plots per subfig: baseline and filter
        # Run the EnKF+model for multiple population sizes (multiple times)
        pop_sizes = [2, 5, 10, 20, 50, 100]
        N = 50
        its = 3600
        assimilation_period = 25
        ensemble_size = 10
        mode = EnsembleKalmanFilterType.STATE

        # Plot facet grid by population size
        # Individual plots are sns.relplot, x=time, y=error
        # Two plots per subfig: baseline and filter

    @staticmethod
    def __run_model(N):
        model_params = {'pop_total': N,
                        'station': 'Grand_Central',
                        'do_print': False}
        m = Model(**model_params)

        # Run model
        for _ in range(m.step_limit):
            m.step()

        try:
            c = len(m.history_wiggle_locs)
            t = m.max_time
            d = {'collisions': c,
                 'time': t,
                 'population_size': N}
            return d
        except:
            print('failure')
            return None


class Processor():
    def __init__(self):
        pass

    @classmethod
    def process_batch(cls, read_time=False, write_time=True):
        """
        Process the output data from a batch of runs.

        Stage 1:
        Consider each file in the results/repeats/ directory.
        For each file:
        1) Derive parameter values from the filename.
        2) Read the results.
        3) Find the means for the forecast, analytis and observations.
        4) Average each over time.
        5) Add to existing results.
        6) If writing then output combined results to json.

        Stage 2:
        1) If reading then read in exsiting results, else follow stage 1.
        2) Convert output data to dataframe.
        3) Produce a collection of heatmaps to summarise results.

        Parameters
        ----------
        read_time : boolean
            Boolean to choose whether to read in existing json of results.
        write_time : boolean
            Boolean to choose whether to write out processed results to json.
        """
        if read_time:
            with open('results/map_data.json') as f:
                output = json.load(f)
        else:
            # Set up link to directory
            results_path = './results/repeats/'
            results_list = listdir(results_path)
            output = list()

            for r in results_list:
                # Derive parameters from filename
                components = r.split('__')

                ap = int(components[0].split('_')[-1])
                es = int(components[1])
                pop_size = int(components[2])
                pre_sigma = components[3].split('.')[0]
                sigma = float(pre_sigma.replace('_', '.'))

                # Read in set of results:
                p = './results/repeats/{0}'.format(r)
                with open(p) as f:
                    d = json.load(f)

                # Reduce to means for forecast, analysis and obs
                forecasts, analyses, observations = cls.process_repeat_results(d)

                # Take mean over time
                forecast = forecasts['mean'].mean()
                analysis = analyses['mean'].mean()
                observation = observations['mean'].mean()

                # Add to output list
                row = {'assimilation_period': ap,
                       'ensemble_size': es,
                       'population_size': pop_size,
                       'std': sigma,
                       'forecast': forecast,
                       'analysis': analysis,
                       'observation': observation}

                output.append(row)

            if write_time:
                with open('results/map_data.json', 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=4)

        data = pd.DataFrame(output)
        Visualiser.make_all_heatmaps(data)

    @staticmethod
    def extract_array(df, var1, var2):
        """
        Function to extract data array pertaining to the variables that we are
        interested in.

        Extract an array of the mean errors with two parameters varying; other
        parameters are kept fixed.
        First define the default values for each of the four possible parameters
        (assimilation period, ensemble size, population size and observation noise
        standard deviation).
        Get the sorted values that each of the chosen parameters take.
        Create an array of the data that fits the above conditions, and convert
        into an array with column indices taking the values of the first parameter
        and the row indices taking the value of the second parameter.

        Parameters
        ----------
        df : pandas dataframe
            A pandas dataframe containing all of the mean errors for each of the
            parameter combinations.
        var1 : string
            Name of the first variable that we want to consider variation with
            respect to.
        var2 : string
            Name of the second variable that we want to consider variation with
            respect to.
        """
        # Define variables to fix and filter
        fixed_values = {'assimilation_period': 20,
                        'ensemble_size': 20,
                        'population_size': 15,
                        'std': 1.5}

        var1_vals = sorted(df[var1].unique())
        var2_vals = sorted(df[var2].unique())
        fix_vars = [x for x in fixed_values.keys() if x not in [var1, var2]]
        print(var1, var1_vals)
        print(var2, var2_vals)

        # Filtering down to specific fixed values
        cond1 = df[fix_vars[0]] == fixed_values[fix_vars[0]]
        cond2 = df[fix_vars[1]] == fixed_values[fix_vars[1]]
        tdf = df[cond1 & cond2]

        # Reformat to array
        a = np.zeros(shape=(len(var1_vals), len(var2_vals)))
        for i, u in enumerate(var1_vals):
            for j, v in enumerate(var2_vals):
                var1_cond = tdf[var1] == u
                var2_cond = tdf[var2] == v
                d = tdf[var1_cond & var2_cond]
                a[i][j] = d['analysis'].values[0]

        output = pd.DataFrame(a, index=var1_vals, columns=var2_vals)
        # output = pd.DataFrame(a.T, index=var2_vals, columns=var1_vals)
        # return a.T, var2_vals, var1_vals
        return output.T

    @classmethod
    def process_repeat_results(cls, results):
        """
        process_repeat_results

        Takes the results of running the enkf repeatedly and restructures it into
        separate data structures for forecasts, analyses and observations.

        Parameters
        ----------
        results : list(list(dict()))
            Each list entry is a list of dictionaries which stores the time-series
            of the forecast, analysis and observations for that realisation.
            Each dictionary contains entries for:
                - time
                - forecast
                - analysis
                - observation
        """
        forecasts = list()
        analyses = list()
        observations = list()
        first = True
        times = list()

        for res in results:
            # Sort results by time
            res = sorted(res, key=lambda k: k['time'])
            forecast = list()
            analysis = list()
            observation = list()
            for r in res:
                if first:
                    times.append(r['time'])
                forecast.append(r['forecast'])
                analysis.append(r['analysis'])
                observation.append(r['obs'])
            first = False
            forecasts.append(forecast)
            analyses.append(analysis)
            observations.append(observation)

        forecasts = cls.make_dataframe(forecasts, times)
        analyses = cls.make_dataframe(analyses, times)
        observations = cls.make_dataframe(observations, times)
        return forecasts, analyses, observations


    @staticmethod
    def make_dataframe(dataset, times):
        """
        make_dataframe

        Make a dataframe from a dataset.
        This requires that the data undergo the following transformations:
            - Convert to numpy array
            - Transpose array
            - Convert array to pandas dataframe
            - Calculate row-mean in new column
            - Add time to data
            - Set time as index

        Parameters
        ----------
        dataset : list(list())
            List of lists containing data.
            Each inner list contains a single time-series.
            The outer list contains a collection of inner lists, each pertaining to
            a realisation of the model.
        times : list-like
            List of times at which data is provided.
        """
        d = pd.DataFrame(np.array(dataset).T)
        m = d.mean(axis=1)
        s = d.std(axis=1)
        up = d.max(axis=1)
        down = d.min(axis=1)
        d['mean'] = m
        d['up_diff'] = up - m
        d['down_diff'] = m - down
        d['sd'] = s
        d['time'] = times
        return d.set_index('time')


class Visualiser():
    def __init__(self):
        pass

    @classmethod
    def make_all_heatmaps(cls, data):
        """
        Make a collection of error heatmaps.

        Use plot_heatmap() to produce heatmaps showing how the mean error varies
        with respect to assimilation period and population size, ensemble size and
        population size, and obsevation standard deviation and population size.

        Parameters
        ----------
        data : pandas dataframe
            A pandas dataframe containing mean errors and values for each of the
            input parameters.
        """
        # plot_heatmap(data, 'assimilation_period', 'ensemble_size')
        cls.plot_heatmap(data, 'assimilation_period', 'population_size')
        # plot_heatmap(data, 'assimilation_period', 'std')
        cls.plot_heatmap(data, 'ensemble_size', 'population_size')
        # plot_heatmap(data, 'ensemble_size', 'std')
        cls.plot_heatmap(data, 'std', 'population_size')

    @staticmethod
    def plot_heatmap(data, var1, var2):
        """
        Plotting a heat map of variation of errors with respect to two variables.

        Extract the appropriate data array from the data.
        Produce a matplotlib contour plot of the variation of the mean error with
        respect to var1 and var2.
        Save as a pdf figure with name based on the two variables.

        Parameters
        ----------
        data : pandas dataframe
            A pandas dataframe in which each row pertains to the error resulting
            from an input set of parameters. Consequently, each row contains the
            mean error, as well as the relevant parameter values.
        var1 : string
            The first variable against which we would like to measure the variation
            of the mean error.
        var2 : string
            The second variable against which we would like to measure the
            variation of the mean error.
        """
        label_dict = {'assimilation_period': 'Assimilation Period',
                      'ensemble_size': 'Ensemble Size',
                      'population_size': 'Population Size',
                      'std': 'Observation Error Standard Deviation'}
        # d, rows, cols = extract_array(data, var1, var2)
        # print(d)
        # print(rows)
        # print(cols)
        # plt.contourf(rows, cols, d)
        d = Processor.extract_array(data, var1, var2)
        print(d)
        plt.contourf(d, levels=10, cmap='PuBu')
        plt.yticks(np.arange(0, len(d.index), 1), d.index)
        plt.xticks(np.arange(0, len(d.columns), 1), d.columns)
        # plt.xticks(np.arange(0, len(cols), 1), cols)
        # plt.yticks(np.arange(0, len(rows), 1), rows)
        plt.xlabel(label_dict[var1])
        plt.ylabel(label_dict[var2])
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./results/figures/{0}_{1}.pdf'.format(var1, var2))
        plt.show()

    @staticmethod
    def plot_results(dataset):
        """
        plot_results

        Plot results for a single dataset (i.e. either forecast, analysis or
        observations). Produces a line graph containing individual lines for each
        realisation (low alpha and dashed), and a line for the mean of the
        realisations (full alpha and bold).

        Parameters
        ----------
        dataset : pandas dataframe
            pandas dataframe of data containing multiple realisations and mean of
            all realisations indexed on time.
        """
        no_plot = ['sd', 'up_diff', 'down_diff']
        colnames = list(dataset)
        plt.figure()
        for col in colnames:
            if col == 'mean':
                plt.plot(dataset[col], 'b-', linewidth=5, label='mean')
            elif col not in no_plot:
                plt.plot(dataset[col], 'b--', alpha=0.25, label='_nolegend_')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('RMSE')
        plt.show()

    @staticmethod
    def plot_all_results(forecast, analysis, observation):
        """
        plot_all_results

        Plot forecast, analysis and observations in one plot.
        Contains three subplots, each one pertaining to one of the datasets.
        Subplots share x-axis and y-axis.

        Parameters
        ----------
        forecast : pandas dataframe
            pandas dataframe of forecast data.
        analysis : pandas dataframe
            pandas dataframe of analysis data.
        observation : pandas dataframe
            pandas dataframe of observation data.
        """
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                          figsize=(8, 12))
        # f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

        no_plot = ['sd', 'up_diff', 'down_diff']

        colnames = list(forecast)
        for col in colnames:
            if col == 'mean':
                ax1.plot(forecast[col], 'b-', linewidth=2, label='forecast mean')
            elif col not in no_plot:
                ax1.plot(forecast[col], 'b--', alpha=0.25, label='_nolegend_')
        ax1.legend(loc='upper left')
        ax1.set_ylabel('RMSE')

        colnames = list(analysis)
        for col in colnames:
            if col == 'mean':
                ax2.plot(analysis[col], 'g-', linewidth=2, label='analysis mean')
            elif col not in no_plot:
                ax2.plot(analysis[col], 'g--', alpha=0.25, label='_nolegend_')
        ax2.legend(loc='upper left')
        ax2.set_ylabel('RMSE')

        colnames = list(observation)
        for col in colnames:
            if col == 'mean':
                ax3.plot(observation[col], 'k-', linewidth=2, label='observation mean')
            elif col not in no_plot:
                ax3.plot(observation[col], 'k--', alpha=0.25, label='_nolegend_')
        ax3.legend(loc='upper left')
        ax3.set_xlabel('time')
        ax3.set_ylabel('RMSE')

        plt.savefig('results/figures/all_results.pdf')

        plt.show()

    @staticmethod
    def plot_with_errors(forecast, analysis, observation):
        """
        Plot results with errors.

        Plot forecast, analysis and observations in one plot.
        Contains three subplots, each one pertaining to one of the datasets.
        Subplots share x-axis and y-axis.
        Each subplot contains a mean line, and a shaded area pertaining to the
        range of the data at each point in model time.

        Parameters
        ----------
        forecast : pandas dataframe
            pandas dataframe of forecast data.
        analysis : pandas dataframe
            pandas dataframe of analysis data.
        observation : pandas dataframe
            pandas dataframe of observation data.
        """
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True,
                                          figsize=(8, 12))

        ax1.plot(forecast['mean'], 'b-', label='forecast mean')
        ax1.fill_between(forecast.index,
                         forecast['mean'] - forecast['down_diff'],
                         forecast['mean'] + forecast['up_diff'],
                         alpha=0.25)
        ax1.legend(loc='upper left')
        ax1.set_ylabel('RMSE')

        ax2.plot(analysis['mean'], 'g-', label='analysis mean')
        ax2.fill_between(analysis.index,
                         analysis['mean'] - analysis['down_diff'],
                         analysis['mean'] + analysis['up_diff'],
                         alpha=0.25)
        ax2.legend(loc='upper left')
        ax2.set_ylabel('RMSE')

        ax3.plot(observation['mean'], 'k-', label='observation mean')
        ax3.fill_between(observation.index,
                         observation['mean'] - observation['down_diff'],
                         observation['mean'] + observation['up_diff'],
                         alpha=0.25)
        ax3.legend(loc='upper left')
        ax3.set_ylabel('RMSE')

        plt.savefig('results/figures/all_with_errors.pdf')

        plt.show()

    @staticmethod
    def plot_model(enkf, title_str, obs=[]):
        """
        Plot base_model, observations and ensemble mean for each agent.
        """
        # Get coords
        base_state = enkf.base_model.get_state(sensor='location')
        base_x, base_y = enkf.separate_coords(base_state)
        mean_x, mean_y = enkf.separate_coords(enkf.state_mean)
        if len(obs) > 0:
            obs_x, obs_y = enkf.separate_coords(obs)
        if enkf.run_vanilla:
            vanilla_x, vanilla_y = enkf.separate_coords(enkf.vanilla_state_mean)

        # Plot agents
        plot_width = 8
        aspect_ratio = enkf.base_model.height / enkf.base_model.width
        plot_height = round(aspect_ratio * plot_width)
        plt.figure(figsize=(plot_width, plot_height))
        plt.xlim(0, enkf.base_model.width)
        plt.ylim(0, enkf.base_model.height)
        # plt.title(title_str)
        plt.scatter(base_x, base_y, label='Ground truth')
        plt.scatter(mean_x, mean_y, marker='x', label='Ensemble mean')
        if len(obs) > 0:
            plt.scatter(obs_x, obs_y, marker='*', label='Observation')
        if enkf.run_vanilla:
            plt.scatter(vanilla_x, vanilla_y, alpha=0.5, label='mean w/o da',
                        color='black')

        # Plot ensemble members
        # for model in self.models:
            # xs, ys = self.separate_coords(model.state)
            # plt.scatter(xs, ys, s=1, color='red')

        # Finish fig
        plt.legend()
        plt.xlabel('x-position')
        plt.ylabel('y-position')
        plt.savefig('./results/figures/{0}.eps'.format(title_str))
        plt.show()

    @staticmethod
    def plot_model2(enkf, title_str, obs=[]):
        """
        Plot base_model and ensemble members for a single agent.
        """
        # List of all plotted coords
        x_coords = list()
        y_coords = list()

        # Get coords
        base_state = enkf.base_model.get_state(sensor='location')
        base_x, base_y = enkf.separate_coords(base_state)
        mean_x, mean_y = enkf.separate_coords(enkf.state_mean)
        if len(obs) > 0:
            obs_x, obs_y = enkf.separate_coords(obs)
        if enkf.run_vanilla:
            vanilla_x, vanilla_y = enkf.separate_coords(enkf.vanilla_state_mean)

        # Plot agents
        plot_width = 8
        aspect_ratio = enkf.base_model.height / enkf.base_model.width
        plot_height = round(aspect_ratio * plot_width)
        plt.figure(figsize=(plot_width, plot_height))
        plt.xlim(0, enkf.base_model.width)
        plt.ylim(0, enkf.base_model.height)
        # plt.title(title_str)
        plt.scatter(base_x[enkf.agent_number],
                    base_y[enkf.agent_number], label='Ground truth')
        plt.scatter(mean_x[enkf.agent_number],
                    mean_y[enkf.agent_number], marker='x', label='Ensemble mean')
        if len(obs) > 0:
            plt.scatter(obs_x[enkf.agent_number],
                        obs_y[enkf.agent_number], marker='*', label='Observation')
        if enkf.run_vanilla:
            plt.scatter(vanilla_x, vanilla_y, alpha=0.5, label='mean w/o da',
                        color='black')

        x_coords.extend([base_x[enkf.agent_number], mean_x[enkf.agent_number],
                         obs_x[enkf.agent_number]])
        y_coords.extend([base_y[enkf.agent_number], mean_y[enkf.agent_number],
                         obs_y[enkf.agent_number]])

        # Plot ensemble members
        for i, model in enumerate(enkf.models):
            state_vector = model.get_state(sensor='location')
            xs, ys = enkf.separate_coords(state_vector)
            if i == 0:
                plt.scatter(xs[enkf.agent_number], ys[enkf.agent_number],
                            s=1, color='red', label='Ensemble members')
            else:
                plt.scatter(xs[enkf.agent_number], ys[enkf.agent_number],
                            s=1, color='red')
            x_coords.append(xs[enkf.agent_number])
            y_coords.append(ys[enkf.agent_number])

        # Finish fig
        plt.legend()
        plt.xlabel('x-position')
        plt.ylabel('y-position')
        plt.xlim(x_coords[0]-5, x_coords[0]+5)
        plt.ylim(y_coords[0]-5, y_coords[0]+5)
        # plt.xlim(min(x_coords)-1, max(x_coords)+1)
        # plt.ylim(min(y_coords)-1, max(y_coords)+1)
        plt.savefig('./results/figures/{0}_single.eps'.format(title_str))
        plt.show()

    @staticmethod
    def plot_model_results(distance_errors, x_errors, y_errors):
        """
        Method to plot the evolution of errors in the filter.
        """
        plt.figure()
        plt.scatter(range(len(distance_errors)), distance_errors,
                    label='$\mu$', s=1)
        plt.scatter(range(len(x_errors)), x_errors, label='$\mu_x$', s=1)
        plt.scatter(range(len(y_errors)), y_errors, label='$\mu_y$', s=1)
        plt.xlabel('Time')
        plt.ylabel('MAE')
        plt.legend()
        plt.savefig('./results/figures/errors.eps')
        plt.show()

    @staticmethod
    def plot_model_results2(errors_da, errors_vanilla):
        """
        Method to plot the evolution of errors for abm with da vs w/o da.
        """
        plt.figure()
        plt.scatter(range(len(errors_da)), errors_da, label='with')
        plt.scatter(range(len(errors_vanilla)), errors_vanilla, label='w/o')
        plt.xlabel('Time')
        plt.ylabel('MAE')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_error_timeseries(enkf, model_params, filter_params, do_save=False):
        results = pd.DataFrame(enkf.metrics)
        plt.figure(figsize=(8, 8))
        plt.plot(results['time'], results['obs'], label='observations')
        plt.plot(results['time'], results['forecast'], label='filter_forecast')
        plt.plot(results['time'], results['analysis'], label='filter_analysis')
        if enkf.run_vanilla:
            plt.plot(results['time'], results['vanilla'], label='vanilla')

        plt.legend()

        if do_save:
            plt.savefig('results/figures/error_timeseries.pdf')
        else:
            plt.show()

        if enkf.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            plt.figure(figsize=(12, 8))
            plt.plot(results['time'], results['exit_accuracy'])
            if do_save:
                plt.savefig('results/figures/exit_accuracy.pdf')
            else:
                plt.show()

    @classmethod
    def plot_forecast_error_timeseries(cls, enkf, model_params, filter_params,
                                       do_save=False, plot_period=True):
        """
        plot_forecast_error_timeseries

        Plot the variation of Ensemble Kalman Filter forecast error over time
        (iterations).
        Additionally, vertical lines may be plotted to indicate timesteps at
        which data has been assimilated.

        Parameters
        ----------
        enkf : EnsembleKalmanFilter class
            An instance of the Ensemble Kalman Filter that has been run and is
            to have its forecast error plotted.
        model_params : dict
            Dictionary of parameters passed to each model in the ensemble.
        filter_params : dict
            Dictionary of parameters dictating the behaviour of the filter.
        do_save : boolean
            Indicates whether or not to save a copy of the plots.
        plot_period : boolean
            Indicates whether or not to plot vertical lines indicating timesteps
            at which data have been assimilated.
        """
        results = pd.DataFrame(enkf.forecast_error)
        plt.figure(figsize=(8, 8))
        plt.scatter(results['time'], results['forecast'], s=0.75)
        plt.xlabel('iteration')
        plt.ylabel('forecast rmse')

        if plot_period:
            period = filter_params['assimilation_period']
            assimilation_ticks = cls.__make_assimilation_ticks(results['time'],
                                                               period)
            for t in assimilation_ticks:
                plt.axvline(t, linestyle='dotted', color='black', alpha=0.25)
        if do_save:
            plt.savefig('results/figures/forecast_timeseries.pdf')
        else:
            plt.show()

    @staticmethod
    def __make_assimilation_ticks(times, period):
        """
        make_assimilation_ticks

        Given the times at which agent states are known and the regularity with
        which data is assimilated, provide a list of the times when the data
        assimilation has taken place.

        Parameters
        ----------
        times : list
            Times at which the model state is known.
        period : int
            The period with which data is assimilated.
        """
        ticks = [time for time in times if time % period == 0]
        return ticks

    @staticmethod
    def plot_endtimes(data):
        for population_size, times in data.items():
            plt.figure(figsize=(8, 8))
            plt.hist(times, bins=20)
            plt.title(f'Population = {population_size}')
            plt.xlabel('Endtime (iterations)')
            plt.ylabel('Frequency')
            plt.savefig(f'results/figures/endtimes_{population_size}.pdf')

    @staticmethod
    def plot_exits(enkf):
        gl = enkf.base_model.gates_locations
        plt.figure()
        for i, g in enumerate(gl):
            plt.scatter(g[0], g[1], label=i)
        plt.legend(loc='center')
        plt.savefig('results/figures/exit_map.pdf')

    @staticmethod
    def plot_exit_accuracy(enkf, model_params):
        pop_size = model_params['pop_total']
        exits = enkf.exits
        agent_exits = [list() for _ in range(pop_size)]
        for t in range(len(exits)):
            for i in range(pop_size):
                agent_exits[i].append(exits[t][i])

        plt.figure()
        for i in range(len(agent_exits)):
            plt.plot(agent_exits[i], label=str(i))
        plt.legend()
        plt.savefig('./results/figures/exits.pdf')
