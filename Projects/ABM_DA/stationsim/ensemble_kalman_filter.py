"""
EnsembleKalmanFilter.py
@author: ksuchak1990
date_created: 19/04/10
A class to represent a general Ensemble Kalman Filter for use with StationSim.
"""

# Imports
from copy import deepcopy as dcopy
from enum import Enum, auto
from filter import Filter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import warnings as warns


# Classes
class EnsembleKalmanFilterType(Enum):
    STATE = auto()
    PARAMETER_EXIT = auto()
    DUAL_EXIT = auto()


class EnsembleKalmanFilter(Filter):
    """
    A class to represent a general EnKF.
    """

    def __init__(self, model, filter_params, model_params, benchmarking=False):
        """
        Initialise the Ensemble Kalman Filter.

        Params:
            model
            filter_params
            model_params

        Returns:
            None
        """
        # Instantiates the base model and starts time at 0
        super().__init__(model, model_params)

        # Filter attributes - outlines the expected params
        self.__assign_filter_defaults()

        # Get filter attributes from params, warn if unexpected attribute
        self.__assign_filter_params(filter_params)

        # Set up ensemble of models
        self.models = self.__set_up_models()
        # Make sure that models have state
        # for m in self.models:
        # if not hasattr(m, 'state'):
        # raise AttributeError("Model has no 'state' attribute.")

        # We're going to need H.T very often, so just do it once and store
        self.H_transpose = self.H.T

        # Make sure that we have a data covariance matrix
        """
        https://arxiv.org/pdf/0901.3725.pdf -
        Covariance matrix R describes the estimate of the error of the data;
        if the random errors in the entries of the data vector d are independent,
        R is diagonal and its diagonal entries are the squares of the standard
        deviation (“error size”) of the error of the corresponding entries of the
        data vector d.
        """
        if not self.data_covariance:
            self.data_covariance = np.diag(self.R_vector)

        # Create placeholders for ensembles
        self.__set_up_ensembles()

        # Errors stats at update steps
        self.metrics = list()
        self.forecast_error = list()

        # Vanilla params
        if self.run_vanilla:
            self.__set_up_baseline()

        self.update_state_ensemble()
        self.update_state_means()

        self.results = list()
#        self.results = [self.state_mean]


        # Agent to plot individually
        self.agent_number = 6

        self.__print_start_summary()
        self.original_state = self.state_mean
        self.exits = [self.state_mean[2*self.population_size:]]

    def __set_up_baseline(self):
        # Ensemble of vanilla models is always 10 (control variable)
        self.vanilla_ensemble_size = 10
        self.vanilla_models = self.__set_up_models(self.vanilla_ensemble_size)
        # self.vanilla_models = [dcopy(self.base_model) for _ in
                               # range(self.vanilla_ensemble_size)]
        self.vanilla_state_mean = None
        self.vanilla_state_ensemble = np.zeros(shape=(self.state_vector_length,
                                                      self.vanilla_ensemble_size))
        self.vanilla_metrics = list()
        self.vanilla_results = list()

    def __set_up_ensembles(self):
        self.state_ensemble = np.zeros(shape=(self.state_vector_length,
                                              self.ensemble_size))
        self.state_mean = None
        self.data_ensemble = np.zeros(shape=(self.data_vector_length,
                                             self.ensemble_size))

    def __print_start_summary(self):
        print('Running Ensemble Kalman Filter...')
        print('max_iterations:\t{0}'.format(self.max_iterations))
        print('ensemble_size:\t{0}'.format(self.ensemble_size))
        print('assimilation_period:\t{0}'.format(self.assimilation_period))
        print('filter_type:\t{0}'.format(self.mode))

    def __assign_filter_params(self, filter_params):
        for k, v in filter_params.items():
            if not hasattr(self, k):
                w = 'EnKF received unexpected {0} attribute.'.format(k)
                warns.warn(w, RuntimeWarning)
            setattr(self, k, v)

        self.n_exits = self.base_model.gates_out
        self.sensor_type = self.sensor_types[self.mode]
        self.error_func = self.error_funcs[self.mode]

    def __assign_filter_defaults(self):
        self.max_iterations = None
        self.ensemble_size = None
        self.assimilation_period = None
        self.state_vector_length = None
        self.population_size = None
        self.data_vector_length = None
        self.H = None
        self.R_vector = None
        self.data_covariance = None
        self.keep_results = False
        self.vis = False
        self.run_vanilla = False
        self.mode = EnsembleKalmanFilterType.STATE
        self.active = True
        self.sensor_types = {EnsembleKalmanFilterType.STATE: 'location',
                             EnsembleKalmanFilterType.DUAL_EXIT: 'loc_exit'}
        self.error_funcs = {EnsembleKalmanFilterType.STATE: self.make_errors,
                            EnsembleKalmanFilterType.DUAL_EXIT: self.make_dual_errors}

    def __set_up_models(self, n=None):
        # Set up ensemble of models
        # Deep copy preserves knowledge of agent origins and destinations
        n = self.ensemble_size if n is None else n
        models = [dcopy(self.base_model) for _ in range(n)]

        if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            for model in models:
                for agent in model.agents:
                    # Randomise the destination of each agent in each model
                    gate_out = self.make_random_destination(model.gates_in,
                                                              model.gates_out,
                                                              agent.gate_in)
                    agent.gate_out = gate_out
                    agent.loc_desire = agent.set_agent_location(agent.gate_out)
        elif self.mode != EnsembleKalmanFilterType.STATE:
            raise ValueError('Filter type not recognised.')
        return models

    def make_random_destination(self, gates_in, gates_out, gate_in):
        # Ensure that their destination is not the same as their origin
        gate_out = np.random.randint(gates_out) + gates_in
        while (gate_out == gate_in or gate_out >= self.n_exits):
            gate_out = np.random.randint(gates_out)
        return gate_out

    def step(self):
        """
        Step the filter forward by one time-step.

        Params:
            data

        Returns:
            None
        """
        # Check if any of the models are active
        self.update_status()
        # Only predict-update if there is at least one active model
        if self.active:
            self.predict()
            self.update_state_ensemble()
            self.update_state_means()

            truth = self.base_model.get_state(sensor=self.sensor_type)

            f = self.error_func(truth, self.state_mean)[0]

            forecast_error = {'time': self.time,
                              'forecast': f}
            self.forecast_error.append(forecast_error)

            if self.time % self.assimilation_period == 0:
                # Construct observations
                data, obs_truth = self.make_data()

                # Plot model state
                self.plot_model_state('before')

                # Update
                self.update(data)
                self.update_state_means()
                self.update_models()

                metrics = forecast_error.copy()
                metrics = self.make_metrics(metrics, truth, obs_truth, data)
                self.metrics.append(metrics)

                if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
                    exits = self.state_mean[2*self.population_size]
                    self.exits.append(exits)

                # Plot posterior
                self.plot_model_state('after')

            # else:
                # self.update_state_mean()
            self.time += 1
            self.results.append(self.state_mean)

            if self.run_vanilla:
                self.vanilla_results.append(self.vanilla_state_mean)

        # print('time: {0}, base: {1}'.format(self.time,
            # self.base_model.pop_active))
        # print('time: {0}, models: {1}'.format(self.time, [m.pop_active for m in
            # self.models]))

    def predict(self):
        """
        Step the model forward by one time-step to produce a prediction.

        Params:

        Returns:
            None
        """
        self.base_model.step()
        for i in range(self.ensemble_size):
            self.models[i].step()
        if self.run_vanilla:
            for i in range(self.vanilla_ensemble_size):
                self.vanilla_models[i].step()

    def update(self, data):
        """
        Update filter with data provided.

        Params:
            data

        Returns:
            None
        """
        if len(data) != self.data_vector_length:
            w = 'len(data)={0}, expected {1}'.format(len(data),
                                                     self.data_vector_length)
            warns.warn(w, RuntimeWarning)
        X = np.zeros(shape=(self.state_vector_length, self.ensemble_size))
        self.update_data_ensemble(data)
        gain_matrix = self.make_gain_matrix()
        diff = self.data_ensemble - self.H @ self.state_ensemble
        X = self.state_ensemble + gain_matrix @ diff
        self.state_ensemble = X

    def make_metrics(self, metrics, truth, obs_truth, data):
        # Calculating prior and likelihood errors
        metrics['obs'] = self.make_errors(obs_truth, data)[0]

        # Analysis error
        if self.mode == EnsembleKalmanFilterType.STATE:
            d, _, _ = self.make_analysis_errors(truth, self.state_mean)
        elif self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            # USE ANALYSIS ERRORS
            d, _, _, e = self.make_analysis_errors(truth, self.state_mean)
            metrics['exit_accuracy'] = e
        metrics['analysis'] = d

        # Vanilla error
        if self.run_vanilla:
            v = self.error_func(truth, self.vanilla_state_mean)[0]
            metrics['vanilla'] = v

        return metrics

    def make_data(self):
        # Construct observations
        obs_truth = self.base_model.get_state(sensor='location')
        noise = np.random.normal(0, self.R_vector, obs_truth.shape)
        data = obs_truth + noise
        return data, obs_truth

    def update_state_ensemble(self):
        """
        Update self.state_ensemble based on the states of the models.
        """
        for i in range(self.ensemble_size):
            state_vector = self.models[i].get_state(sensor=self.sensor_type)
            self.state_ensemble[:, i] = state_vector
        if self.run_vanilla:
            for i in range(self.vanilla_ensemble_size):
                state_vector = self.vanilla_models[i].get_state(self.sensor_type)
                self.vanilla_state_ensemble[:, i] = state_vector

    def update_state_means(self):
        self.state_mean = self.update_state_mean(self.state_ensemble)
        if self.run_vanilla:
            self.vanilla_state_mean = self.update_state_mean(self.vanilla_state_ensemble)

    def update_state_mean(self, state_ensemble):
        """
        Update self.state_mean based on the current state ensemble.
        """
        state_mean = np.mean(state_ensemble, axis=1)

        # Round exits if they are in the state vectors
        if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            destinations = state_mean[2*self.population_size:]
            destinations = self.round_destinations(destinations,
                                                     self.n_exits)
            state_mean[2*self.population_size:] = destinations

        return state_mean

    def update_data_ensemble(self, data):
        """
        Create perturbed data vector.
        I.e. a replicate of the data vector plus normal random n-d vector.
        R - data covariance; this should be either a number or a vector with
        same length as the data.
        """
        x = np.zeros(shape=(len(data), self.ensemble_size))
        for i in range(self.ensemble_size):
            x[:, i] = data + np.random.normal(0, self.R_vector, len(data))

        self.data_ensemble = x

    def update_models(self):
        """
        Update individual model states based on state ensemble.
        """
        for i in range(self.ensemble_size):
            state_vector = self.state_ensemble[:, i]

            # Update based on enkf type
            if self.mode == EnsembleKalmanFilterType.STATE:
                self.models[i].set_state(state_vector, sensor='location')
            elif self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
                # Update locations
                locations = state_vector[:2*self.population_size]
                self.models[i].set_state(locations, sensor='location')

                # Update destinations
                destinations = state_vector[2*self.population_size:]
                destinations = self.round_destinations(destinations,
                                                         self.n_exits)
                self.models[i].set_state(destinations, sensor='exit')

    def make_ensemble_covariance(self):
        """
        Create ensemble covariance matrix.
        """
        a = self.state_ensemble @ np.ones(shape=(self.ensemble_size, 1))
        b = np.ones(shape=(1, self.ensemble_size))
        A = self.state_ensemble - 1/self.ensemble_size * a @ b
        return 1/(self.ensemble_size - 1) * A @ A.T

    def make_gain_matrix(self):
        """
        Create kalman gain matrix.
        """
        """
        Version from Gillijns, Barrero Mendoza, etc.
        # Find state mean and data mean
        data_mean = np.mean(self.data_ensemble, axis=1)

        # Find state error and data error matrices
        state_error = np.zeros(shape=(self.state_vector_length,
                                      self.ensemble_size))
        data_error = np.zeros(shape=(self.data_vector_length,
                                     self.ensemble_size))
        for i in range(self.ensemble_size):
            state_error[:, i] = self.state_ensemble[:, i] - self.state_mean
            data_error[:, i] = self.data_ensemble[:, i] - data_mean
        P_x = 1 / (self.ensemble_size - 1) * state_error @ state_error.T
        P_xy = 1 / (self.ensemble_size - 1) * state_error @ data_error.T
        P_y = 1 / (self.ensemble_size -1) * data_error @ data_error.T
        K = P_xy @ np.linalg.inv(P_y)
        return K
        """
        """
        More standard version
        """
        C = np.cov(self.state_ensemble)
        state_covariance = self.H @ C @ self.H_transpose
        diff = state_covariance + self.data_covariance
        return C @ self.H_transpose @ np.linalg.inv(diff)

    @staticmethod
    def separate_coords(arr):
        """
        Function to split a flat array into xs and ys.
        Assumes that xs and ys alternate.
        i.e.
        [x0, y0, x1, y1] -> ([x0, x1], [y0, y1])
        """
        if len(arr) % 2 != 0:
            raise ValueError('Please provide an array of even length.')
        return arr[::2], arr[1::2]

    @staticmethod
    def pair_coords(arr1, arr2):
        if len(arr1) != len(arr2):
            raise ValueError('Both arrays should be the same length.')
        results = list()
        for i in range(len(arr1)):
            results.append(arr1[i])
            results.append(arr2[i])
        return results

    def process_results(self):
        """
        Method to process ensemble results, comparing against truth.
        Calculate x-error and y-error for each agent at each timestep,
        average over all agents, and plot how average errors vary over time.
        """
        x_mean_errors = list()
        y_mean_errors = list()
        distance_mean_errors = list()
        base_states = self.base_model.history_state
        truth = [np.ravel(x) for x in base_states]
        without = list()

        for i, result in enumerate(self.results):
            distance_error, x_error, y_error = self.make_errors(result, truth[i])
            x_mean_errors.append(np.mean(x_error))
            y_mean_errors.append(np.mean(y_error))
            distance_mean_errors.append(np.mean(distance_error))

        for i, result in enumerate(self.vanilla_results):
            wo, j, k = self.make_errors(result, truth[i])
            without.append(np.mean(wo))

        # if self.vis:
            # self.plot_results(distance_mean_errors, x_mean_errors, y_mean_errors)
            # self.plot_results2(distance_mean_errors, without)

        # Save results to csv if required
        if self.keep_results:
            df = pd.DataFrame({'distance_errors': distance_mean_errors,
                               'x_errors': x_mean_errors,
                               'y_errors': y_mean_errors})
            self.save_results(df)

        # Comparing rmse
        metrics = pd.DataFrame(self.metrics)
        if self.vis:
            plt.figure()
            plt.plot(metrics['time'], metrics['obs'], label='Observation')
            plt.plot(metrics['time'], metrics['forecast'], label='Forecast')
            plt.plot(metrics['time'], metrics['analysis'], label='Analysis')
            plt.xlabel('Time')
            plt.ylabel('RMSE')
            plt.legend()
            plt.savefig('./results/rmse_comparison.eps')
            plt.show()

    def save_results(self, data):
        """
        Utility method to save the results of a filter run.
        """
        population_size = self.base_model.params['pop_total']
        general_path = './results/enkf_{0}.csv'
        params_string = '{0}_{1}_{2}_{3}'.format(self.max_iterations,
                                                 self.assimilation_period,
                                                 self.ensemble_size,
                                                 population_size)
        data_path = general_path.format(params_string)
        print('Writing filter results to {0}.'.format(data_path))
        data.to_csv(data_path, index=False)

    @classmethod
    def make_errors(cls, truth, result):
        """
        Method to calculate x-errors and y-errors
        """
        x_result, y_result = cls.separate_coords(result)
        x_truth, y_truth = cls.separate_coords(truth)

        d, x, y = cls.calculate_rmse(x_truth, y_truth, x_result, y_result)

        return d, x, y

    @staticmethod
    def make_distance_error(x_error, y_error):
        agent_distances = np.sqrt(np.square(x_error) + np.square(y_error))
        return np.mean(agent_distances)

    def separate_coords_exits(self, state_vector):
        x = state_vector[:self.population_size]
        y = state_vector[self.population_size: 2 * self.population_size]
        e = state_vector[2 * self.population_size:]
        return x, y, e

    def make_dual_errors(self, truth, result):
        x_result, y_result, exit_result = self.separate_coords_exits(result)
        x_truth, y_truth, exit_truth = self.separate_coords_exits(truth)

        d, x, y = self.calculate_rmse(x_truth, y_truth, x_result, y_result)
        exit_accuracy = accuracy_score(exit_truth, exit_result)

        return d, x, y, exit_accuracy

    def make_analysis_errors(self, truth, result):
        if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            return self.make_dual_errors(truth, result)
        elif self.mode == EnsembleKalmanFilterType.STATE:
            return self.make_errors(truth, result)

    @classmethod
    def calculate_rmse(cls, x_truth, y_truth, x_result, y_result):
        """
        Method to calculate the rmse over all agents for a given data set at a
        single time-step.
        """
        x_error = np.mean(np.abs(x_result - x_truth))
        y_error = np.mean(np.abs(y_result - y_truth))
        distance_error = cls.make_distance_error(x_error, y_error)

        return distance_error, x_error, y_error

    def update_status(self):
        """
        update_status

        Update the status of the filter to indicate whether it is active.
        The filter should be active if and only if there is at least 1 active
        model in the ensemble.
        """
        model_statuses = [m.status == 1 for m in self.models]
        self.active = any(model_statuses)

    @classmethod
    def round_destinations(cls, destinations, n_destinations):
        vfunc = np.vectorize(cls.round_destination)
        return vfunc(destinations, n_destinations)

    @staticmethod
    def round_destination(destination, n_destinations):
        dest = int(round(destination))
        return dest % n_destinations

    def plot_model_state(self, when):
        plt.figure()
        # Plot exits
        gl = self.base_model.gates_locations
        for i, g in enumerate(gl):
            plt.scatter(g[0], g[1], c='black', alpha=0.5)

        # Plot ensembles
        for model in self.models:
            for agent in model.agents:
                plt.scatter(agent.location[0], agent.location[1],
                            c='red', s=0.1)

        plt.savefig(f'./results/figures/{when}_{self.time}.pdf')
