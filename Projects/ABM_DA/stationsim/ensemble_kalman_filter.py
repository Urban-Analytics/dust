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
import statistics
from typing import List, Tuple
import warnings as warns


# Classes
class EnsembleKalmanFilterType(Enum):
    STATE = auto()
    PARAMETER_EXIT = auto()
    DUAL_EXIT = auto()


class AgentIncluder(Enum):
    BASE = auto()
    MODE_EN = auto()


class GateEstimator(Enum):
    NO_ESTIMATE = auto()
    ROUNDING = auto()
    ANGLE = auto()


class EnsembleKalmanFilter(Filter):
    """
    A class to represent a general EnKF.
    """

    def __init__(self, model, filter_params, model_params,
                 filtering=True, benchmarking=False):
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

        self.__assign_filter_defaults()
        self.__assign_filter_params(filter_params)
        self.filtering = filtering
        self.run_vanilla = benchmarking

        if self.filtering:
            # Filter attributes - outlines the expected params

            # Get filter attributes from params, warn if unexpected attribute

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
            Covariance matrix R describes the estimate of the error of the
            data; if the random errors in the entries of the data vector d are
            independent, R is diagonal and its diagonal entries are the squares
            of the standard deviation (“error size”) of the error of the
            corresponding entries of the data vector d.
            """
            if not self.data_covariance:
                self.data_covariance = np.diag(self.R_vector)

            # Create placeholders for ensembles
            self.__set_up_ensembles()

        # Vanilla params
        if self.run_vanilla:
            self.vanilla_ensemble_size = filter_params['vanilla_ensemble_size']
            self.__set_up_baseline()

        if self.inclusion is None:
            self.mean_func = np.mean
        else:
            self.mean_func = self.get_population_mean

        # Errors stats at update steps
        self.metrics = list()
        self.forecast_error = list()

        self.update_state_ensemble()
        self.update_state_means()

        self.results = list()
#        self.results = [self.state_mean]

        # Agent to plot individually
        self.agent_number = 6

        self.__print_start_summary()
        # self.original_state = self.state_mean
        # self.exits = [self.state_mean[2*self.population_size:]]

    # --- Set-up Methods --- #
    def __set_up_baseline(self) -> None:
        # Ensemble of vanilla models is always 10 (control variable)
        self.vanilla_models = self.__set_up_models(self.vanilla_ensemble_size)
        # self.vanilla_models = [dcopy(self.base_model) for _ in
        #                        range(self.vanilla_ensemble_size)]
        self.vanilla_state_mean = None
        s = (self.state_vector_length, self.vanilla_ensemble_size)
        self.vanilla_state_ensemble = np.zeros(shape=s)
        self.vanilla_metrics = list()
        self.vanilla_results = list()

    def __set_up_ensembles(self) -> None:
        self.state_ensemble = np.zeros(shape=(self.state_vector_length,
                                              self.ensemble_size))
        self.state_mean = None
        self.data_ensemble = np.zeros(shape=(self.data_vector_length,
                                             self.ensemble_size))

    def __print_start_summary(self) -> None:
        print('Running Ensemble Kalman Filter...')
        print(f'max_iterations:\t{self.max_iterations}')
        print(f'ensemble_size:\t{self.ensemble_size}')
        print(f'assimilation_period:\t{self.assimilation_period}')
        print(f'pop_size:\t{self.population_size}')
        print(f'filter_type:\t{self.mode}')
        print(f'inclusion_type:\t{self.inclusion}')
        print(f'ensemble_errors:\t{self.ensemble_errors}')

    def __assign_filter_defaults(self) -> None:
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
        self.error_normalisation = None
        self.inclusion = None
        self.active = True
        self.gate_estimator = GateEstimator.NO_ESTIMATE
        self.sensor_types = {EnsembleKalmanFilterType.STATE: 'location',
                             EnsembleKalmanFilterType.DUAL_EXIT: 'loc_exit'}
        self.error_funcs = {
            EnsembleKalmanFilterType.STATE: self.make_errors,
            EnsembleKalmanFilterType.DUAL_EXIT: self.make_dual_errors
        }
        self.ensemble_errors = False

    def __assign_filter_params(self, filter_params: dict) -> None:
        for k, v in filter_params.items():
            if not hasattr(self, k):
                w = f'EnKF received unexpected attribute ({k}).'
                warns.warn(w, RuntimeWarning)
            if v is not None:
                setattr(self, k, v)

        self.n_exits = self.base_model.gates_out
        self.sensor_type = self.sensor_types[self.mode]
        self.error_func = self.error_funcs[self.mode]

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

    def make_random_destination(self, gates_in: int,
                                gates_out: int, gate_in: int) -> int:
        # Ensure that their destination is not the same as their origin
        gate_out = np.random.randint(gates_out) + gates_in
        while (gate_out == gate_in or gate_out >= self.n_exits):
            gate_out = np.random.randint(gates_out)
        return gate_out

    # --- Filter step --- #
    def step(self) -> None:
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

            # truth = self.base_model.get_state(sensor=self.sensor_type)
            # state_mean = self.state_mean

            # if self.inclusion is not None:
            #     # Get statuses by which to filter state vector
            #     statuses = self.get_state_vector_statuses(vector_mode=self.mode)

            #     # Filter ground truth vector and state mean vector
            #     truth = self.filter_vector(truth, statuses)
            #     state_mean = self.filter_vector(self.state_mean, statuses)

            # f = self.error_func(truth, state_mean)[0]

            truth = self.base_model.get_state(sensor=self.sensor_type)
            f = self.get_forecast_error(truth)
            forecast_error = {'time': self.time,
                              'forecast': f}
            self.forecast_error.append(forecast_error)

            data = None

            prior = self.state_mean.copy()
            prior_ensemble = self.state_ensemble.copy()

            if self.time % self.assimilation_period == 0:
                # Construct observations
                obs_truth = self.base_model.get_state(sensor='location')
                data = self.make_data(obs_truth, self.R_vector)

                # Plot model state
                if self.vis:
                    self.plot_model_state('before')

                # Update
                self.update(data)
                self.update_state_means()
                self.update_models()

                metrics = forecast_error.copy()
                metrics = self.make_metrics(metrics, truth, obs_truth, data)
                self.metrics.append(metrics)

                if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
                    exits = self.state_mean[2 * self.population_size]
                    self.exits.append(exits)

                # Plot posterior
                if self.vis:
                    self.plot_model_state('after')

                # Collect state information for vis
                result = self.collect_results(obs_truth, prior, data,
                                              prior_ensemble)
                self.results.append(result)

            # else:
                # self.update_state_mean()

            if self.run_vanilla:
                self.vanilla_results.append(self.vanilla_state_mean)

            self.time += 1
            # self.results.append(self.state_mean)

        # print('time: {0}, base: {1}'.format(self.time,
            # self.base_model.pop_active))
        # print('time: {0}, models: {1}'.format(self.time, [m.pop_active for
        #                                                   m in self.models]))

    def baseline_step(self) -> None:
        # Check if any of the models are active
        self.update_status()
        # Only predict-update if there is at least one active model
        if self.active:
            self.predict()
            self.update_state_ensemble()
            self.update_state_means()

            truth = self.base_model.get_state(sensor=self.sensor_type)

            f = self.error_func(truth, self.vanilla_state_mean)

            forecast_error = {'time': self.time,
                              'forecast': f}
            self.forecast_error.append(forecast_error)

            self.time += 1

            self.vanilla_results.append(self.vanilla_state_mean)

    def predict(self) -> None:
        """
        Step the model forward by one time-step to produce a prediction.

        Params:

        Returns:
            None
        """
        self.base_model.step()
        if self.filtering:
            for i in range(self.ensemble_size):
                self.models[i].step()
        if self.run_vanilla:
            for i in range(self.vanilla_ensemble_size):
                self.vanilla_models[i].step()

    def update(self, data) -> None:
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
        gain_matrix = self.make_gain_matrix(self.state_ensemble,
                                            self.data_covariance,
                                            self.H,
                                            self.H_transpose)
        diff = self.data_ensemble - self.H @ self.state_ensemble
        X = self.state_ensemble + gain_matrix @ diff
        self.state_ensemble = X

    # --- Error calculation --- #
    @classmethod
    def get_x_y_diffs(cls, truth: np.ndarray,
                      results: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Separate coords
        x_result, y_result = cls.separate_coords(results)
        x_truth, y_truth = cls.separate_coords(truth)

        # Calculate diffs
        x_diffs = np.abs(x_result - x_truth)
        y_diffs = np.abs(y_result - y_truth)

        return x_diffs, y_diffs

    def get_forecast_error(self, truth):
        state_mean = self.state_mean.copy()

        if self.inclusion is not None:
            # Get statuses by which to filter state vector
            statuses = self.get_state_vector_statuses(vector_mode=self.mode)

            # Filter ground truth vector and state mean vector
            truth = self.filter_vector(truth, statuses)
            state_mean = self.filter_vector(self.state_mean, statuses)

        error = self.error_func(truth, state_mean)
        return error

    def make_metrics(self, metrics: dict, truth: np.ndarray,
                     obs_truth: np.ndarray, data: np.ndarray) -> dict:
        """
        Calculate error metrics.

        Given a dictionary containing the prior error, calculate the posterior
        and observation error.
        Additionally, if baseline models are being run, calculate error for them
        too.
        Uses self.make_obs_error() to calculate observation errors,
        self.make_analysis_errors to calculate posterior error, and
        self.error_func to calculate baseline error when necessary.
        Posterior may include exit accuracy if running a DUAL_EXIT filter.

        Parameters
        ----------
        metrics : dict
            Dictionary or errors, already contains prior error.
        truth : np.ndarray
            Vector of true agent locations taken from the base model using the
            sensor type used for model states.
        obs_truth : np.ndarray
            Vector of true agent locations taken from the base model using
            'location' sensor.
        data : np.ndarray
            Vector of observations of agent locations.
        """
        state_mean = self.state_mean.copy()
        if self.run_vanilla:
            vanilla_state_mean = self.vanilla_state_mean.copy()
        if self.inclusion is not None:
            # Get statuses by which to filter state vector
            statuses = self.get_state_vector_statuses(vector_mode=self.mode)

            # Filter ground truth vector and state mean vector
            truth = self.filter_vector(truth, statuses)
            obs_truth = self.filter_vector(obs_truth, statuses)
            data = self.filter_vector(data, statuses)
            state_mean = self.filter_vector(state_mean, statuses)
            if self.run_vanilla:
                vanilla_state_mean = self.filter_vector(vanilla_state_mean,
                                                        statuses)

        # Calculating prior and likelihood errors
        metrics['obs'] = self.make_errors(obs_truth, data)

        # Analysis error
        if self.mode == EnsembleKalmanFilterType.STATE:
            d = self.make_errors(truth, state_mean)
        elif self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            # USE ANALYSIS ERRORS
            d, e = self.make_dual_errors(truth, state_mean)
            metrics['exit_accuracy'] = e
        else:
            general_message = 'Please provide an appropriate filter type.'
            spec_message = f'{self.mode} is not an acceptable filter type.'
            s = f'{general_message} {spec_message}'
            raise ValueError(s)
        metrics['analysis'] = d

        if self.ensemble_errors:
            ensemble_errors = self.get_ensemble_errors(truth)
            metrics.update(ensemble_errors)

        # Vanilla error
        if self.run_vanilla:
            v = self.error_func(truth, vanilla_state_mean)
            metrics['baseline'] = v

        return metrics

    def get_ensemble_errors(self, truth: np.ndarray) -> dict:
        ensemble_errors = dict()
        for i in range(self.ensemble_size):
            model_error = self.get_ensemble_error(truth, i)
            ensemble_errors[f'analysis_{i}'] = model_error
        return ensemble_errors

    def get_ensemble_error(self, truth: np.ndarray, model_idx) -> float:
        state = self.state_ensemble[:, model_idx]
        if self.inclusion is not None:
            statuses = self.get_state_vector_statuses(vector_mode=self.mode)
            state = self.filter_vector(state, statuses)
        return self.make_errors(truth, state)

    def make_errors(self, truth, result) -> float:
        """
        Calculate errors.

        Given a vector of truth agent locations and a vector of estimated agent
        locations, calculate the x-errors, y-errors and distance errors.
        Vectors are separated in to x-coords and y-coords and then passed to
        self.calculate_rmse() which returns the relevant errors.

        Parameters
        ----------
        truth : np.ndarray
            Vector of true agent locations.
        result : np.ndarray
            Vector of estimated agent locations.

        Returns
        -------
        float
            distance error
        """
        x_diffs, y_diffs = self.get_x_y_diffs(truth, result)
        d = self.make_distance_error(x_diffs, y_diffs)

        return d

    def make_dual_errors(self, truth: np.ndarray,
                         result: np.ndarray) -> Tuple[float, float]:
        """
        Calculate errors for dual filter.

        Given a vector of truth agent locations and exits and a vector of
        estimated agent locations and exits, calculate the x-errors, y-errors
        distance errors and exit accuracy.
        Vectors are separated in to x-coords, y-coords and estimated exits.
        x-y coords are passed to self.calculate_rmse() which returns distance
        errors, x-errors and y-errors.
        Exit accuracy is evaluated using sklearn's accuracy_score().

        Parameters
        ----------
        truth : np.ndarray
            Vector of true agent locations and exits.
        result : np.ndarray
            Vector of estimated agent locations and exits.

        Returns
        -------
        Tuple[float, float, float, float]:
            distance error, x-error, y-error, exit accuracy.
        """
        x_result, y_result, exit_result = self.separate_coords_exits(result)
        x_truth, y_truth, exit_truth = self.separate_coords_exits(truth)

        d, _, _ = self.calculate_rmse(x_truth, y_truth, x_result, y_result)
        exit_accuracy = accuracy_score(exit_truth, exit_result)

        return d, exit_accuracy

    def make_distance_error(self, x_error: np.ndarray,
                            y_error: np.ndarray) -> float:
        """
        Calculate Euclidean distance errors.

        Given an array of x-errors and y-errors, calculate the Euclidean
        distance errors.
        This involves calculating the distances based on
            d_i = sqrt(x_i ** 2 + y_i ** 2)

        The mean distance is calculated based on the vector of d_i using the
        mean calculation function allocated to self.mean_func().

        Parameters
        ----------
        x_error : np.ndarray
            Vector of x-errors.
        y_error : np.ndarray
            Vector of y-errors.

        Returns
        -------
        float:
            Mean distance error.
        """
        agent_distances = np.sqrt(np.square(x_error) + np.square(y_error))
        return np.mean(agent_distances)

    def make_analysis_errors(self, truth: np.ndarray, result: np.ndarray):
        """
        Calculate analysis/posterior error.

        Given a vector of true agent states and a vector of estimated agent
        states, calculate the posterior error.
        This method acts as a wrapper for self.make_errors() and
        self.make_dual_errors() choosing the appropriate method as appropriate
        depending on whether filter is performing location state estimation or
        location-gate estimation.

        Parameters
        ----------
        truth : np.ndarray
            Vector of true agent states.
        result : np.ndarray
            Vector of representing the posterior agent states.
        """
        if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            return self.make_dual_errors(truth, result)
        elif self.mode == EnsembleKalmanFilterType.STATE:
            return self.make_errors(truth, result)
        else:
            general_message = 'Please provide an appropriate filter type.'
            spec_message = f'{self.mode} is not an acceptable filter type.'
            s = f'{general_message} {spec_message}'
            raise ValueError(s)

    def make_obs_error(self, truth: np.ndarray, result: np.ndarray) -> float:
        """
        Calculate observation error.

        Given a vector of true agent locations and a vector of observed agent
        locations, calculate the observation error.
        This involves:
            - Separating states into x-coords and y-coords,
            - Calculating the differences/errors between truth and result,
            - Calculating the Euclidean distaces,
            - Finding the mean distance.

        Parameters
        ----------
        truth : np.ndarray
            Vector of true agent states.
        result : np.ndarray
            Vector of observed agent states.

        Returns
        -------
        float:
            Average distance error.
        """
        agent_distances = self.make_errors(truth, result)
        return np.mean(agent_distances)

    def calculate_rmse(self, x_truth: np.ndarray,
                       y_truth: np.ndarray,
                       x_result: np.ndarray,
                       y_result: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate RMSE, i.e. root mean squared error.

        Given vectors of true x-coords, true y-coords, estimated x-coords and
        estimated y-coords, calculate x-errors, y-errors and Euclidean distance
        errors.
        This involves:
            - Calculating the differences between truth and estimate for x- and
              y-coords,
            - Calculating the average x- and y-errors using the appropriate mean
              function allocated to self.mean_func(),
            - Calculating the average distance error using
              self.make_distance_error().

        Parameters
        ----------
        x_truth : np.ndarray
            Vector of true x-positions of agents.
        y_truth : np.ndarray
            Vector of true y-positions of agents.
        x_result : np.ndarray
            Vector of estimated x-positions of agents.
        y_result : np.ndarray
            Vector of estimated y-positions of agents.

        Returns
        -------
        Tuple[float, float, float]:
            Distance error, x-error, y-error.
        """
        """
        Method to calculate the rmse over all agents for a given data set at a
        single time-step.
        """
        x_diffs = np.abs(x_result - x_truth)
        y_diffs = np.abs(y_result - y_truth)
        x_error = np.mean(x_diffs)
        y_error = np.mean(y_diffs)
        distance_error = self.make_distance_error(x_diffs, y_diffs)

        return distance_error, x_error, y_error

    def get_population_mean(self, arr: np.ndarray) -> float:
        """
        Calculate population mean of vector.

        Calculate mean of a vector, dividing the sum by the number of active
        agents in the population instead of the length of the vector (i.e. the
        population size).

        Parameters
        ----------
        arr : np.ndarray
            Vector of quantities to be averaged.

        Returns
        -------
        float:
            Population mean quantity.
        """
        n = self.get_n_active_agents()
        pm = 0 if n == 0 else np.sum(arr) / n
        return pm

    @classmethod
    def base_inclusion_error(cls, result: np.ndarray, truth: np.ndarray,
                             base_statuses: list) -> float:
        x_result, y_result = cls.separate_coords(result)
        x_truth, y_truth = cls.separate_coords(truth)

        x_result = cls.filter_vector(x_result, base_statuses)
        y_result = cls.filter_vector(y_result, base_statuses)
        x_truth = cls.filter_vector(x_truth, base_statuses)
        y_truth = cls.filter_vector(y_truth, base_statuses)

        x_diff = x_result - x_truth
        y_diff = y_result - y_truth

        agent_distances = np.sqrt(np.square(x_diff) + np.square(y_diff))
        return np.mean(agent_distances)

    def get_mean_error(self, results: np.ndarray,
                       truth: np.ndarray) -> float:
        diff = np.abs(results - truth)
        return self.mean_func(diff)

    @staticmethod
    def filter_vector(vector: np.ndarray, statuses: list) -> np.ndarray:
        """
        Filter a vector of quantities.

        Filter a vector of quantities based on a list of statuses.
        The output only contains elements from the original vector for which
        the respective element in statuses is non-zero.

        Parameters
        ----------
        vector : np.ndarray
            Vector of quantities to be filtered.
        statuses : list
            List of statuses.

        Returns
        -------
        np.ndarray:
            Filtered vector of quantities.
        """
        # Ensure same number of quantities as statuses
        assert len(vector) == len(statuses)

        # List comprehension to filter vector based on respective statuses
        output = [vector[i] for i in range(len(vector)) if statuses[i]]

        return np.array(output)

    # --- Update methods --- #
    def update_state_ensemble(self) -> None:
        """
        Update self.state_ensemble based on the states of the models.
        """
        st = self.sensor_type
        if self.filtering:
            for i in range(self.ensemble_size):
                state_vector = self.models[i].get_state(sensor=st)
                self.state_ensemble[:, i] = state_vector
        if self.run_vanilla:
            for i in range(self.vanilla_ensemble_size):
                state_vector = self.vanilla_models[i].get_state(st)
                self.vanilla_state_ensemble[:, i] = state_vector

    def update_state_means(self) -> None:
        if self.filtering:
            self.state_mean = self.update_state_mean(self.state_ensemble)
        if self.run_vanilla:
            state_ensemble = self.vanilla_state_ensemble
            self.vanilla_state_mean = self.update_state_mean(state_ensemble)

    def update_state_mean(self, state_ensemble: np.ndarray) -> np.ndarray:
        """
        Update self.state_mean based on the current state ensemble.
        """
        state_mean = np.mean(state_ensemble, axis=1)

        # Round exits if they are in the state vectors
        if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            destinations = state_mean[2 * self.population_size:]
            destinations = self.round_destinations(destinations,
                                                   self.n_exits)
            state_mean[2 * self.population_size:] = destinations

        return state_mean

    def update_data_ensemble(self, data) -> None:
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

    def update_models(self) -> None:
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
                locations = state_vector[:2 * self.population_size]
                self.models[i].set_state(locations, sensor='location')

                # Update destinations
                destinations = state_vector[2 * self.population_size:]
                destinations = self.round_destinations(destinations,
                                                       self.n_exits)
                self.models[i].set_state(destinations, sensor='exit')

    def update_status(self) -> None:
        """
        update_status

        Update the status of the filter to indicate whether it is active.
        The filter should be active if and only if there is at least 1 active
        model in the ensemble.
        """
        if self.filtering:
            m_statuses = [m.status == 1 for m in self.models]
        else:
            m_statuses = [False]

        if self.run_vanilla:
            vanilla_m_statuses = [m.status == 1 for m in self.vanilla_models]
        else:
            vanilla_m_statuses = [False]

        self.active = any(m_statuses) or any(vanilla_m_statuses)

    # --- Filter step helper methods --- #
    @classmethod
    def make_data(cls, obs_truth, R_vector):
        # Construct observations
        noise = cls.make_noise(obs_truth.shape, R_vector)
        data = obs_truth + noise
        return data

    @staticmethod
    def make_noise(shape, R_vector, seed=None):
        # Seed for test reproducibility
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(0, R_vector, shape)

    # def make_ensemble_covariance(self) -> np.array:
    #     """
    #     Create ensemble covariance matrix.
    #     """
    #     a = self.state_ensemble @ np.ones(shape=(self.ensemble_size, 1))
    #     b = np.ones(shape=(1, self.ensemble_size))
    #     A = self.state_ensemble - 1/self.ensemble_size * a @ b
    #     return 1/(self.ensemble_size - 1) * A @ A.T

    @staticmethod
    def make_gain_matrix(state_ensemble: np.ndarray,
                         data_covariance: np.ndarray,
                         H, H_transpose) -> np.ndarray:
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
        C = np.cov(state_ensemble)
        state_covariance = H @ (C @ H_transpose)
        total = state_covariance + data_covariance
        K = C @ (H_transpose @ np.linalg.inv(total))
        return K

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
    def pair_coords(arr1, arr2) -> list:
        if len(arr1) != len(arr2):
            raise ValueError('Both arrays should be the same length.')
        results = list()
        for i in range(len(arr1)):
            results.append(arr1[i])
            results.append(arr2[i])
        return results

    def get_n_active_agents(self) -> int:
        if self.inclusion == AgentIncluder.BASE:
            n = self.base_model.pop_active
        else:
            n_active_ensemble = [model.pop_active for model in self.models]
            if self.inclusion == AgentIncluder.MODE_EN:
                n = statistics.mode(n_active_ensemble)
            else:
                raise ValueError('Unrecognised AgentIncluder type')
        return n

    def separate_coords_exits(self,
                              state_vector: np.ndarray) -> Tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray]:
        x = state_vector[:self.population_size]
        y = state_vector[self.population_size: 2 * self.population_size]
        e = state_vector[2 * self.population_size:]
        return x, y, e

    @classmethod
    def round_destinations(cls, destinations, n_destinations):
        vfunc = np.vectorize(cls.round_destination)
        return vfunc(destinations, n_destinations)

    @staticmethod
    def round_destination(destination: float, n_destinations: int) -> int:
        dest = int(round(destination))
        return dest % n_destinations

    def get_agent_statuses(self) -> List[bool]:
        if self.inclusion is None or self.inclusion == AgentIncluder.BASE:
            # List of booleans indicating whether agents are active
            statuses = [agent.status == 1 for agent in self.base_model.agents]

        elif self.inclusion == AgentIncluder.MODE_EN:
            en_statuses = [list() for _ in range(self.population_size)]

            # Get list of statuses for each agent
            for model in self.models:
                for j, agent in enumerate(model.agents):
                    en_statuses[j].append(agent.status == 1)

            # Assigned status is the modal status across the ensemble of models
            statuses = [statistics.mode(l) for l in en_statuses]

        else:
            s = f'Inclusion type ({self.inclusion}) not recognised.'
            raise ValueError(s)
        return statuses

    def get_state_vector_statuses(self, vector_mode) -> List[bool]:
        # Repeat statuses each agent
        # Twice for STATE, i.e. x-y coords
        # Three times for DUAL_EXIT, i.e. x-y-exit

        agent_statuses = self.get_agent_statuses()

        # Define whether to repeat statuses 2 or 3 times
        n = 3 if vector_mode == EnsembleKalmanFilterType.DUAL_EXIT else 2

        statuses = list()
        for x in agent_statuses:
            statuses.extend([x for _ in range(n)])

        return statuses

    def set_base_statuses(self, base_statuses: List[int]) -> None:
        assert len(base_statuses) == len(self.base_model.agents)

        for i, agent in enumerate(self.base_model.agents):
            agent.status = base_statuses[i]

        return None

    def set_ensemble_statuses(self, ensemble_statuses: List[List[int]]) -> None:
        assert len(ensemble_statuses) == len(self.models)

        for i, model in enumerate(self.models):
            assert len(model.agents) == len(ensemble_statuses[i])

            for j, agent in enumerate(model.agents):
                agent.status = ensemble_statuses[i][j]

        return None

    # --- Data processing --- #
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
            distance_error, x_error, y_error = self.__make_errors(result,
                                                                  truth[i])
            x_mean_errors.append(np.mean(x_error))
            y_mean_errors.append(np.mean(y_error))
            distance_mean_errors.append(np.mean(distance_error))

        for i, result in enumerate(self.vanilla_results):
            wo, j, k = self.make_errors(result, truth[i])
            without.append(np.mean(wo))

        # if self.vis:
        #     self.plot_results(distance_mean_errors,
        #                       x_mean_errors, y_mean_errors)
        #     self.plot_results2(distance_mean_errors, without)

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

    def save_results(self, data) -> None:
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

    def plot_model_state(self, when: str) -> None:
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
        plt.close()

    def make_animation(self) -> None:
        # Check that filter has finished running
        if self.active:
            raise ValueError('Please wait until filter has finished running')

        """
        Required data:
            - mean state vector
            - state vectors from ensemble member models
            - base model state vector
            - observation state vector

        Need to make sure that these line up with each other, i.e we have the
        same number of frames to plot for each of the state vectors
        """

    def make_base_destinations_vector(self) -> np.ndarray:
        destinations = list()
        for agent in self.base_model.agents:
            destinations.append(agent.loc_desire)

        return np.ravel(destinations)

    def make_base_origins_vector(self) -> np.ndarray:
        origins = list()
        for agent in self.base_model.agents:
            if agent.status == 0:
                origins.append(np.array([0, 0]))
            else:
                origins.append(agent.loc_start)

        return np.ravel(origins)

    def collect_results(self, obs_truth, prior, data, prior_ensemble):
        result = {'time': self.time,
                  'ground_truth': obs_truth,
                  'prior': prior,
                  'posterior': self.state_mean.copy()}
        result['observation'] = data
        result['destination'] = self.make_base_destinations_vector()
        result['origin'] = self.make_base_origins_vector()

        for i in range(self.ensemble_size):
            result[f'prior_{i}'] = prior_ensemble[:, i]
            result[f'posterior_{i}'] = self.state_ensemble[:, i].copy()

        if self.run_vanilla:
            result['baseline'] = self.vanilla_state_mean

        return result
