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
from math import atan2, pi
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


class ExitRandomisation(Enum):
    NONE = auto()
    BY_AGENT = auto()
    ALL_RANDOM = auto()
    ADJACENT = auto()


class Inflation(Enum):
    NONE = auto()
    MULTIPLICATIVE = auto()
    ADDITIVE = auto()


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

        if self.gate_estimator == GateEstimator.ANGLE:
            self.__set_angle_estimation_defaults()

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
        self.exit_randomisation = ExitRandomisation.NONE
        self.n_adjacent = None
        self.inflation = Inflation.NONE
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
        if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            self.exits = list()

    def __set_up_models(self, n=None):
        # Set up ensemble of models
        # Deep copy preserves knowledge of agent origins and destinations
        n = self.ensemble_size if n is None else n
        models = [dcopy(self.base_model) for _ in range(n)]

        if self.exit_randomisation == ExitRandomisation.BY_AGENT:
            gates_in = self.base_model.gates_in
            gates_out = self.base_model.gates_out
            for i, agent in enumerate(self.base_model.agents):
                gate_out = self.make_random_destination(gates_in,
                                                        gates_out,
                                                        agent.gate_in)
                target = agent.set_agent_location(gate_out)
                for model in models:
                    model.agents[i].gate_out = gate_out
                    model.agents[i].loc_desire = target
        elif self.exit_randomisation == ExitRandomisation.ALL_RANDOM:
            gates_in = self.base_model.gates_in
            gates_out = self.base_model.gates_out
            for model in models:
                for agent in model.agents:
                    gate_out = self.make_random_destination(gates_in,
                                                            gates_out,
                                                            agent.gate_in)
                    agent.gate_out = gate_out
                    agent.loc_desire = agent.set_agent_location(gate_out)
        elif self.exit_randomisation == ExitRandomisation.ADJACENT:
            for i, agent in enumerate(self.base_model.agents):
                gate_out = agent.gate_out

                for model in models:
                    # Set up offsets
                    lower_offset = -self.n_adjacent
                    upper_offset = self.n_adjacent
                    offset = np.random.randint(lower_offset, upper_offset)
                    # Apply offset to gate_out
                    model_gate_out = gate_out + offset
                    model_gate_out = model_gate_out % self.base_model.gates_out
                    # Create location
                    target = agent.set_agent_location(model_gate_out)
                    # Assign to agent
                    model.agents[i].gate_out = model_gate_out
                    model.agents[i].loc_desire = target

        # if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
        #     for model in models:
        #         for agent in model.agents:
        #             # Randomise the destination of each agent in each model
        #             gate_out = self.make_random_destination(model.gates_in,
        #                                                     model.gates_out,
        #                                                     agent.gate_in)
        #             agent.gate_out = gate_out
        #             agent.loc_desire = agent.set_agent_location(gate_out)
        # elif self.mode != EnsembleKalmanFilterType.STATE:
        #     raise ValueError('Filter type not recognised.')
        return models

    def __set_angle_estimation_defaults(self):
        self.__set_corner_angles()
        self.__set_gates_edge_angles()

    def __set_corner_angles(self):
        self.corners = {'top_left': (0, self.base_model.height),
                        'top_right': (self.base_model.width,
                                      self.base_model.height),
                        'bottom_left': (0, 0),
                        'bottom_right': (self.base_model.width, 0)}

        self.model_centre = (self.base_model.width / 2,
                             self.base_model.height / 2)

        self.corner_angles = dict()
        for corner, location in self.corners.items():
            angle = self.get_angle(self.model_centre, location)
            self.corner_angles[corner] = angle

    def __set_gates_edge_angles(self):
        n_gates = len(self.base_model.gates_locations)

        self.gate_angles = dict()
        gate_angles = list()

        for gate_number in range(n_gates):
            gate_loc = self.base_model.gates_locations[gate_number]
            gate_width = self.base_model.gates_width[gate_number]
            edge_loc_1, edge_loc_2 = self.__get_gate_edge_locations(gate_loc,
                                                                    gate_width)
            edge_1, edge_2 = self.__get_gate_edge_angles(self.model_centre,
                                                         edge_loc_1,
                                                         edge_loc_2)
            self.gate_angles[gate_number] = (edge_1, edge_2)
            gate_angles.extend([(edge_1, edge_loc_1),
                                (edge_2, edge_loc_2)])

        unique_angle_list = list(set(gate_angles))
        sorted_angle_list = sorted(unique_angle_list,
                                   key=lambda info: info[0],
                                   reverse=True)
        self.unique_gate_angles = [x[0] for x in sorted_angle_list]
        self.unique_gate_edges = [x[1] for x in sorted_angle_list]

        in_gate_idx = [0, 2, 4, 6, 8, 10, 12, 14, 15, 16, 17, 19]
        self.in_gate_idx = {idx: i for i, idx in enumerate(in_gate_idx)}
        out_gate_idx = [x for x in range(19) if x not in self.in_gate_idx]
        self.out_gate_idx = set(out_gate_idx)
        # Note equivalence between edge index and gate index
        self.edge_to_gate = {0: 0, 1: 1, 2: 1,
                             3: 2, 4: 2, 5: 3,
                             6: 3, 7: 4, 8: 4,
                             9: 5, 10: 5, 11: 6,
                             12: 6, 13: 7, 17: 10,
                             18: 0}

    def __get_gate_edge_locations(self, gate_loc, gate_width):
        wd = gate_width / 2

        if(gate_loc[0] == 0):
            edge_loc_1 = (0, gate_loc[1] + wd)
            edge_loc_2 = (0, gate_loc[1] - wd)
        elif(gate_loc[0] == self.base_model.width):
            edge_loc_1 = (self.base_model.width, gate_loc[1] + wd)
            edge_loc_2 = (self.base_model.width, gate_loc[1] - wd)
        elif(gate_loc[1] == 0):
            edge_loc_1 = (gate_loc[0] + wd, 0)
            edge_loc_2 = (gate_loc[0] - wd, 0)
        elif(gate_loc[1] == self.base_model.height):
            edge_loc_1 = (gate_loc[0] + wd, self.base_model.height)
            edge_loc_2 = (gate_loc[0] - wd, self.base_model.height)
        else:
            raise ValueError(f'Invalid gate location: {gate_loc}')
        return edge_loc_1, edge_loc_2

    def __get_gate_edge_angles(self, centre_loc, edge_loc_1, edge_loc_2):
        edge_angle_1 = self.get_angle(centre_loc, edge_loc_1)
        edge_angle_2 = self.get_angle(centre_loc, edge_loc_2)

        return edge_angle_1, edge_angle_2

    def make_random_destination(self, gates_in: int,
                                gates_out: int, gate_in: int) -> int:
        # Ensure that their destination is not the same as their origin
        gate_out = np.random.randint(gates_out) + gates_in
        while (gate_out == gate_in or gate_out >= self.n_exits):
            gate_out = np.random.randint(gates_out)
        return gate_out

    @staticmethod
    def get_angle(vector_tail: Tuple[float, float],
                  vector_head: Tuple[float, float]) -> float:
        x_diff = vector_head[0] - vector_tail[0]
        y_diff = vector_head[1] - vector_tail[1]
        angle = atan2(y_diff, x_diff)
        return angle

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

            if self.get_n_active_agents() > 0:
                truth = self.base_model.get_state(sensor=self.sensor_type)
                f = self.get_forecast_error(truth)
                forecast_error = {'time': self.time,
                                  'forecast': f}
                self.forecast_error.append(forecast_error)

                data = None

                prior = self.state_mean.copy()
                prior_ensemble = self.state_ensemble.copy()

                if self.time % self.assimilation_period == 0:
                    self.assimilation_step(truth, data, forecast_error, prior,
                                           prior_ensemble)
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

    def assimilation_step(self, truth, data, forecast_error, prior,
                          prior_ensemble):
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
        # if len(data) != self.data_vector_length:
        #     w = 'len(data)={0}, expected {1}'.format(len(data),
        #                                              self.data_vector_length)
        #     warns.warn(w, RuntimeWarning)
        X = np.zeros(shape=(self.state_vector_length, self.ensemble_size))

        if data.ndim == 1:
            self.update_data_ensemble(data)
        elif data.ndim == 2:
            self.data_ensemble = data
        else:
            raise ValueError(f'Data has unexpected ndim: {data.ndim}')
        self.gain_matrix = self.make_gain_matrix(self.state_ensemble,
                                                 self.data_covariance,
                                                 self.H,
                                                 self.H_transpose)
        diff = self.data_ensemble - self.H @ self.state_ensemble
        X = self.state_ensemble + self.gain_matrix @ diff
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

        if self.gate_estimator == GateEstimator.ANGLE:
            state_mean = self.convert_vector_angle_to_gate(state_mean)

        if self.inclusion is not None:
            n_active = self.get_n_active_agents()

            # Get statuses by which to filter state vector
            statuses = self.get_state_vector_statuses(vector_mode=self.mode)

            # Filter ground truth vector and state mean vector
            truth = self.filter_vector(truth, statuses)
            state_mean = self.filter_vector(state_mean, statuses)
        else:
            n_active = self.population_size

        if self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            error, _ = self.error_func(truth, state_mean, n_active)
        elif self.mode == EnsembleKalmanFilterType.STATE:
            error = self.error_func(truth, state_mean)
        else:
            raise ValueError(f'Filter type: {self.mode}')
        return error

    def make_metrics(self, metrics: dict, truth: np.ndarray,
                     obs_truth: np.ndarray, data: np.ndarray) -> dict:
        """
        Calculate error metrics.

        Given a dictionary containing the prior error, calculate the posterior
        and observation error.
        Additionally, if baseline models are being run, calculate error for
        them too.
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

        if self.gate_estimator == GateEstimator.ANGLE:
            state_mean = self.convert_vector_angle_to_gate(state_mean)

        if self.run_vanilla:
            vanilla_state_mean = self.vanilla_state_mean.copy()

        if self.inclusion is None:
            n_active = self.population_size
        else:
            n_active = self.get_n_active_agents()

        if self.inclusion is not None:
            # Get statuses by which to filter state vector
            obs_mode = EnsembleKalmanFilterType.STATE

            model_statuses = self.get_state_vector_statuses(self.mode)
            obs_statuses = self.get_state_vector_statuses(obs_mode)

            # Filter ground truth vector and state mean vector
            truth = self.filter_vector(truth, model_statuses)
            obs_truth = self.filter_vector(obs_truth, obs_statuses)
            data = self.filter_vector(data, obs_statuses)
            state_mean = self.filter_vector(state_mean, model_statuses)
            if self.run_vanilla:
                vanilla_state_mean = self.filter_vector(vanilla_state_mean,
                                                        model_statuses)

        # Calculating prior and likelihood errors
        metrics['obs'] = self.make_errors(obs_truth, data)

        # Analysis error
        if self.mode == EnsembleKalmanFilterType.STATE:
            d = self.make_errors(truth, state_mean)
        elif self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            # USE ANALYSIS ERRORS
            d, e = self.make_dual_errors(truth, state_mean, n_active)
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
            if self.mode == EnsembleKalmanFilterType.STATE:
                v = self.error_func(truth, state_mean)
            elif self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
                # USE ANALYSIS ERRORS
                v, _ = self.error_func(truth, state_mean, n_active)
            metrics['baseline'] = v

        return metrics

    def convert_vector_angle_to_gate(self, state: np.ndarray) -> np.ndarray:
        locs = state[: 2 * self.population_size]
        angles = state[2 * self.population_size:]
        gates, _ = self.construct_state_from_angles(angles)

        new_state = np.concatenate((locs, gates))
        return new_state

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

    def make_dual_errors(self, truth: np.ndarray, result: np.ndarray,
                         n_active: int) -> Tuple[float, float]:
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
        x_result, y_result, exit_result = self.separate_coords_exits(n_active,
                                                                     result)
        x_truth, y_truth, exit_truth = self.separate_coords_exits(n_active,
                                                                  truth)

        d, _, _ = self.calculate_rmse(x_truth, y_truth, x_result, y_result)
        # print(self.time)
        # print('truth', truth)
        # print('exit t', exit_truth)
        # print('res', result)
        # print('exit r', exit_result)
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
            - Calculating the average x- and y-errors using the appropriate
              mean function allocated to self.mean_func(),
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
        if self.mode == EnsembleKalmanFilterType.STATE:
            st = 'location'
        elif self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            if self.gate_estimator == GateEstimator.ROUNDING:
                # Returns x, y, g
                st = 'loc_exit'
            elif self.gate_estimator == GateEstimator.ANGLE:
                # Returns x, y, g, d_x, d_y
                st = 'enkf_gate_angle'
            else:
                st = 'loc_exit'
        else:
            raise ValueError(f'Unrecognised mode: {self.mode}')
        if self.filtering:
            for i in range(self.ensemble_size):
                state = self.models[i].get_state(sensor=st)
                processed_state = self.process_state_vector(state)
                self.state_ensemble[:, i] = processed_state
        if self.run_vanilla:
            for i in range(self.vanilla_ensemble_size):
                state = self.vanilla_models[i].get_state(sensor=st)
                processed_state = self.process_state_vector(state)
                self.vanilla_state_ensemble[:, i] = processed_state

    def process_state_vector(self, state):
        if self.mode == EnsembleKalmanFilterType.STATE:
            return state
        elif self.mode == EnsembleKalmanFilterType.DUAL_EXIT:
            if self.gate_estimator != GateEstimator.ANGLE:
                return state
            else:
                # state is x, y, g, d_x, d_y
                state = np.array(state)
                locations = state[:2 * self.population_size]
                destinations = state[3 * self.population_size:]

                # Set up empty array to fill with angles
                angles = np.zeros(self.population_size)

                for i in range(self.population_size):
                    loc = (destinations[i],
                           destinations[self.population_size + i])
                    angles[i] = self.get_angle(self.model_centre, loc)

                reduced_state = np.concatenate((locations, angles))
                return reduced_state
        else:
            raise ValueError(f'Unrecognised mode: {self.mode}')

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
            if self.gate_estimator == GateEstimator.ROUNDING:
                destinations = state_mean[2 * self.population_size:]
                destinations = self.round_destinations(destinations,
                                                       self.n_exits)
                state_mean[2 * self.population_size:] = destinations
            elif self.gate_estimator == GateEstimator.ANGLE:
                angles = state_mean[2 * self.population_size:]
                angles = self.mod_angles(angles)
                state_mean[2 * self.population_size:] = angles

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
                if self.gate_estimator == GateEstimator.ROUNDING:
                    # Update locations
                    locations = state_vector[:2 * self.population_size]

                    # Update destinations
                    destinations = state_vector[2 * self.population_size:]
                    destinations = self.round_destinations(destinations,
                                                           self.n_exits)
                    x = np.concatenate((locations, destinations))
                    self.models[i].set_state(x, sensor='loc_exit')
                elif self.gate_estimator == GateEstimator.ANGLE:
                    locations = state_vector[:2 * self.population_size]
                    angles = state_vector[2 * self.population_size:]
                    # Make sure that we have exactly the correct number of
                    # angles
                    assert len(angles) == self.population_size
                    gates, gate_locs = self.construct_state_from_angles(angles)
                    x = np.concatenate((locations, gates, gate_locs))
                    self.models[i].set_state(x, sensor='enkf_gate_angle')
                else:
                    s = f'Gate estimator no recognised: {self.gate_estimator}'
                    raise ValueError(s)

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

    def make_gain_matrix(self, state_ensemble: np.ndarray,
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

        if self.inflation == Inflation.MULTIPLICATIVE:
            C = self.inflation_rate * C
        elif self.inflation == Inflation.NONE:
            C = C
        elif self.inflation == Inflation.ADDITIVE:
            raise NotImplementedError('Additive inflation not implemented')
        else:
            raise ValueError(f'Unrecognised inflation: {self.inflation}')

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
        return sum(self.get_agent_statuses())
        # if self.inclusion == AgentIncluder.BASE:
        #     n = self.base_model.pop_active
        # else:
        #     n_active_ensemble = [model.pop_active for model in self.models]
        #     if self.inclusion == AgentIncluder.MODE_EN:
        #         n = statistics.mode(n_active_ensemble)
        #     else:
        #         raise ValueError('Unrecognised AgentIncluder type')
        # return n

    def separate_coords_exits(self, n_active: int,
                              state_vector: np.ndarray) -> Tuple[np.ndarray,
                                                                 np.ndarray,
                                                                 np.ndarray]:
        x = state_vector[: n_active]
        y = state_vector[n_active: 2 * n_active]
        e = state_vector[2 * n_active:]
        return x, y, e

    @classmethod
    def convert_alternating_to_sequential(cls, vector):
        assert len(vector) % 2 == 0
        a, b = cls.separate_coords(vector)
        return a + b

    @staticmethod
    def standardise(vector, top, bottom):
        # Find midpoint of range
        midpoint = (top + bottom) / 2

        # Apply shift
        shift_vector = vector - midpoint

        # Calculate shifted top
        shift_top = top - midpoint

        # Scale by shifted top
        standard_vector = shift_vector / shift_top
        return standard_vector


    @staticmethod
    def unstandardise(vector, top, bottom):
        midpoint = (top + bottom) / 2
        shift_top = top - midpoint

        unstandard_vector = vector * shift_top
        unshift_vector = unstandard_vector + midpoint

        return unshift_vector


    @classmethod
    def round_destinations(cls, destinations, n_destinations):
        """
        Vectorize the round_destination() method and apply it to an array of
        destinations.

        Take an array of estimated destinations, and apply the
        round_destination() method to each entry. This is achieved by
        vectorising the method.

        Parameters
        ----------
        destinations : np.ndarray
            Array of estimated destinations
        n_destinations : int
            Number of exits in environment
        """
        vfunc = np.vectorize(cls.round_destination)
        return vfunc(destinations, n_destinations)

    @staticmethod
    def round_destination(destination: float, n_destinations: int) -> int:
        """
        Round estimated destination numbers to whole number values.

        Take a list of estimated destination numbers, and round them to the
        nearest integer value. Where the estimated value is greater than number
        of gates in the environment, apply clock-face/modulo arithmetic to take
        the remainder of the gate number divided by the number of gates, i.e. if
        there are 11 gates and the estimated gate number is 13, we would apply
        the modulo operator to get the remained of dividing 13 by 11 - 2.

        Parameters
        ----------
        destination : float
            Estimated gate number.
        n_destinations : int
            Number of potential exit gates to which an agent could head.

        Returns
        -------
        int:
            Gate number
        """
        dest = int(round(destination))
        return dest % n_destinations

    @staticmethod
    def mod_angles(angles: np.ndarray) -> np.ndarray:
        positive_angles = angles + pi
        bounded_angles = np.mod(positive_angles, 2 * pi)
        output_angles = bounded_angles - pi

        return output_angles

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
            statuses = [statistics.mode(status) for status in en_statuses]

        else:
            s = f'Inclusion type ({self.inclusion}) not recognised.'
            raise ValueError(s)
        return statuses

    def get_state_vector_statuses(self, vector_mode) -> List[bool]:
        # Repeat statuses each agent
        # Twice for STATE, i.e. x-y coords
        # Three times for DUAL_EXIT, i.e. x-y-exit

        agent_statuses = self.get_agent_statuses()
        statuses = list()

        if vector_mode == EnsembleKalmanFilterType.DUAL_EXIT:
            n = 3

            for _ in range(n):
                statuses.extend(agent_statuses)

            return statuses

        elif vector_mode == EnsembleKalmanFilterType.STATE:
            n = 2

            for x in agent_statuses:
                statuses.extend([x for _ in range(n)])

            return statuses
        else:
            raise ValueError(f'Unrecognised filter type: {vector_mode}')

        # Define whether to repeat statuses 2 or 3 times
        # n = 3 if vector_mode == EnsembleKalmanFilterType.DUAL_EXIT else 2

    def set_base_statuses(self, base_statuses: List[int]) -> None:
        assert len(base_statuses) == len(self.base_model.agents)

        for i, agent in enumerate(self.base_model.agents):
            agent.status = base_statuses[i]

        return None

    def set_ensemble_statuses(self,
                              ensemble_statuses: List[List[int]]) -> None:
        assert len(ensemble_statuses) == len(self.models)

        for i, model in enumerate(self.models):
            assert len(model.agents) == len(ensemble_statuses[i])

            for j, agent in enumerate(model.agents):
                agent.status = ensemble_statuses[i][j]

        return None

    def construct_state_from_angles(self, angles) -> Tuple[np.ndarray,
                                                           np.ndarray]:
        """
        Construct a state vector of gate numbers and locations given a list of
        gate angles.

        Take a list of gates - one per agent in the population - and for each
        angle identify the gate number and destination location to which the
        agent will head. Use this information to construct a state vector.

        Parameters
        ----------
        angles : iterable
            List of angles pertaining to the agent population.

        Returns
        -------
        Tuple[np.ndarray,
                                                                   np.ndarray]:
            Numpy array of gate numbers and numpy array of gate locations.
        """
        locations = np.zeros(2 * self.population_size)
        gates = np.zeros(self.population_size)
        for i, angle in enumerate(angles):
            loc, gate = self.get_destination_angle(angle, gate_out=True)
            gates[i] = gate
            locations[i] = loc[0]
            locations[self.population_size + i] = loc[1]

        return gates, locations

    def get_destination_angle(self, angle: float, gate_out: bool = False):

        # If a location is provided then raise an error,
        # We haven't considered this case yet
        # if location is not None:
        #     raise ValueError('Method not implemented for specified locations')

        # location = (self.base_model.width / 2, self.base_model.height / 2)

        insertion_idx = self.bisect_left_reverse(angle,
                                                 self.unique_gate_angles)
        if insertion_idx in self.in_gate_idx:
            # Get gate from dict based on insertion idx
            g = self.in_gate_idx[insertion_idx] % self.base_model.gates_out
            # Use agent method to randomly allocate location along gate
            destination = self.base_model.agents[0].set_agent_location(g)
        elif insertion_idx in self.out_gate_idx:
            # Get index of nearest gate edge
            edge_idx = self.round_target_angle(angle, insertion_idx)
            g = self.edge_to_gate[edge_idx] % self.base_model.gates_out
            # Use index to get location of gate edge
            destination = self.unique_gate_edges[edge_idx]
        else:
            raise ValueError(f'Unrecognised insertion index: {insertion_idx}')

        # TODO describe what happens when angle falls exactly on a gate edge

        if gate_out:
            return destination, g
        else:
            return destination

    def round_target_angle(self, angle: float, insertion_idx: int) -> int:
        """
        Identify index of adjacent gate edge.

        Given an angle (radians) and its insertion index, identify which
        adjacent gate edge index it should be rounded to.

        Parameters
        ----------
        angle : float
            Initial angle.
        insertion_idx : int
            Index at which given angle would be inserted into gate edges.

        Returns
        -------
        int:
            Adjacent index to which it is rounded.
        """
        adjacent_idx = (insertion_idx - 1, insertion_idx)
        adjacent_angles = (self.unique_gate_angles[adjacent_idx[0]],
                           self.unique_gate_angles[adjacent_idx[1]])
        diff_0 = abs(angle - adjacent_angles[0])
        diff_1 = abs(angle - adjacent_angles[1])

        if diff_0 < diff_1:
            return adjacent_idx[0]
        elif diff_1 < diff_0:
            return adjacent_idx[1]
        else:
            return np.random.choice(adjacent_idx)

    @staticmethod
    def bisect_left_reverse(element, iterable) -> int:
        """
        Bisect-left with a reverse-sorted list.

        Given a iterable that is reverse-sorted, i.e. in descending order, find
        the index at which the element would be inserted.

        Parameters
        ----------
        element : numeric
            Element to be inserted into the iterable.
        iterable : iterable-type
            Iterable into which element would be inserted.

        Returns
        -------
        int:
            Index of insertion.
        """
        # Make sure iterable is reverse sorted
        assert iterable == sorted(iterable, reverse=True)
        for i, x in enumerate(iterable):
            if element >= x:
                return i
        return len(iterable)

    @staticmethod
    def is_in_gate_angles(angle: float,
                          edge_angles: Tuple[float, float]) -> bool:
        """
        Check if angle is inside a given gate.

        Given an angle (radians), check if it falls between the two edge angles
        provided.

        Parameters
        ----------
        angle : float
            Angle to check.
        edge_angles : Tuple[float, float]
            Edge angles of a gate.

        Returns
        -------
        bool:
            Indicator of whether angle is within gate.
        """
        return angle >= min(edge_angles) and angle <= max(edge_angles)

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
