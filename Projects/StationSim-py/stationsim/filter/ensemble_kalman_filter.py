"""
EnsembleKalmanFilter.py
@author: ksuchak1990
date_created: 19/04/10
A class to represent a general Ensemble Kalman Filter for use with StationSim.
"""
# Imports
from copy import deepcopy as dcopy
import warnings as warns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filter import Filter


# Classes
class EnsembleKalmanFilter(Filter):
    """
    A class to represent a general EnKF.
    """
    def __init__(self, model, filter_params, model_params):
        """
        Initialise the Ensemble Kalman Filter.

        Params:
            model
            filter_params
            model_params

        Returns:
            None
        """
        # Call parent constructor
        # Instantiates the base model and starts time at 0
        super().__init__(model, model_params)

        # Filter attributes - outlines the expected params
        self.max_iterations = None
        self.ensemble_size = None
        self.assimilation_period = None
        self.state_vector_length = None
        self.data_vector_length = None
        self.H = None
        self.R_vector = None
        self.data_covariance = None
        self.keep_results = False
        self.vis = False

        # Get filter attributes from params, warn if unexpected attribute
        for k, v in filter_params.items():
            if not hasattr(self, k):
                w = 'EnKF received unexpected {0} attribute.'.format(k) 
                warns.warn(w, RuntimeWarning)
            setattr(self, k, v)

        # Set up ensemble of models
        self.models = [dcopy(self.base_model) for _ in range(self.ensemble_size)]

        # Make sure that models have state
        for m in self.models:
            if not hasattr(m, 'state'):
                raise AttributeError("Model has no 'state' attribute.")

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
        self.state_ensemble = np.zeros(shape=(self.state_vector_length,
                                              self.ensemble_size))
        self.state_mean = None
        self.data_ensemble = np.zeros(shape=(self.data_vector_length,
                                             self.ensemble_size))

        self.update_state_ensemble()
        self.update_state_mean()
        self.results = list()
#        self.results = [self.state_mean]
        print('Running Ensemble Kalman Filter...')
        print('max_iterations:\t{0}'.format(self.max_iterations))
        print('ensemble_size:\t{0}'.format(self.ensemble_size))
        print('assimilation_period:\t{0}'.format(self.assimilation_period))

    def step(self):
        """
        Step the filter forward by one time-step.

        Params:
            data

        Returns:
            None
        """
        self.predict()
        self.update_state_ensemble()
        self.update_state_mean()
        if self.time % self.assimilation_period == 0:
            if self.vis:
                self.plot_model('before_update_{0}'.format(self.time))
            truth = self.base_model.state_history[-1]
            noise = np.random.normal(0, self.R_vector, truth.shape)
            data = truth + noise
            self.update(data)
            self.update_models()
            self.update_state_mean()
            if self.vis:
                self.plot_model('after_update_{0}'.format(self.time))
        else:
            self.update_state_mean()
        self.time += 1
        self.results.append(self.state_mean)

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
        gain_matrix = self.make_gain_matrix()
        self.update_data_ensemble(data)
        diff = self.data_ensemble - self.H @ self.state_ensemble
        X = self.state_ensemble + gain_matrix @ diff
        self.state_ensemble = X

    def update_state_ensemble(self):
        """
        Update self.state_ensemble based on the states of the models.
        """
        for i in range(self.ensemble_size):
            self.state_ensemble[:, i] = self.models[i].state

    def update_state_mean(self):
        """
        Update self.state_mean based on the current state ensemble.
        """
        self.state_mean = np.mean(self.state_ensemble, axis=1)

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
            self.models[i].state = self.state_ensemble[:, i]

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
        """
        return arr[::2], arr[1::2]

    def plot_model(self, title_str):
        """
        Plot base_model and ensemble members.
        """
        # Get coords
        base_x, base_y = self.separate_coords(self.base_model.state)
        mean_x, mean_y = self.separate_coords(self.state_mean)

        # Plot agents
        plt.figure(figsize=(8, 8))
        plt.xlim(0, self.base_model.width)
        plt.ylim(0, self.base_model.height)
        plt.title(title_str)
        plt.scatter(base_x, base_y, label='truth')
        plt.scatter(mean_x, mean_y, alpha=0.5, label='ensemble mean')

        # Plot ensemble members
        for model in self.models:
            xs, ys = self.separate_coords(model.state)
            plt.scatter(xs, ys, s=1, color='red')

        # Finish fig
        plt.legend()
        plt.savefig('./results/{0}.png'.format(title_str))
        plt.show()

    def process_results(self):
        """
        Method to process ensemble results, comparing against truth.
        Calculate x-error and y-error for each agent at each timestep,
        average over all agents, and plot how average errors vary over time.
        """
        x_mean_errors = list()
        y_mean_errors = list()
        distance_mean_errors = list()
        truth = self.base_model.state_history

        for i, result in enumerate(self.results):
            distance_error, x_error, y_error = self.make_errors(result, truth[i])
            x_mean_errors.append(np.mean(x_error))
            y_mean_errors.append(np.mean(y_error))
            distance_mean_errors.append(np.mean(distance_error))

        if self.vis:
            self.plot_results(distance_mean_errors, x_mean_errors, y_mean_errors)

        # Save results to csv if required
        if self.keep_results:
            df = pd.DataFrame({'distance_errors': distance_mean_errors,
                               'x_errors': x_mean_errors,
                               'y_errors': y_mean_errors,})
            self.save_results(df)

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
    def make_errors(cls, result, truth):
        """
        Method to calculate x-errors and y-errors
        """
        x_result, y_result = cls.separate_coords(result)
        x_truth, y_truth = cls.separate_coords(truth)

        x_error = np.abs(x_result - x_truth)
        y_error = np.abs(y_result - y_truth)
        distance_error = np.sqrt(np.square(x_error) + np.square(y_error))

        return distance_error, x_error, y_error

    @staticmethod
    def plot_results(distance_errors, x_errors, y_errors):
        """
        Method to plot the evolution of errors in the filter.
        """
        plt.figure()
        plt.scatter(range(len(distance_errors)), distance_errors,
                    label='$\mu$', s=1)
        plt.scatter(range(len(x_errors)), x_errors, label='$\mu_x$', s=1)
        plt.scatter(range(len(y_errors)), y_errors, label='$\mu_y$', s=1)
        plt.xlabel('Time')
        plt.ylabel('Mean absolute error')
        plt.legend()
        plt.savefig('./results/errors.png')
        plt.show()
