"""
Script to run an implementation of the Ensemble Kalman Filter (EnKF).
@author: ksuchak1990
last_modified: 19/04/08
"""
# Imports
import warnings as warns
import numpy as np

# Classes
class EnsembleKalmanFilter:
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
        self.time = 0

        # Ensure that model has correct attributes
        # Should probably make sure that step is callable too
        if not is_good_model(model):
            raise AttributeError('Model does not have required attributes')

        # Filter attributes - outlines the expected params
        self.max_iterations = None
        self.ensemble_size = None
        self.state_vector_length = None
        self.data_vector_length = None
        self.H = None
        self.R_vector = None
        self.data_covariance = None

        # Get filter attributes from params, warn if unexpected attribute
        for k, v in filter_params.items():
            if not hasattr(self, k):
                w = 'EnKF received unexpected {0} attribute.'.format(k) 
                warns.warn(w, RuntimeWarning)
            setattr(self, k, v)

        # Set up ensemble of models
        self.models = [model(model_params) for _ in range(self.ensemble_size)]

        # Make sure that models have state
        for m in self.models:
            if not hasattr(m, 'state'):
                raise AttributeError("Model has no 'state' attribute.")

        # We're going to need H.T very often, so just do it once and store
        self.H_transpose = self.H.T

        # Make sure that we have a data covariance matrix
        """
        https://arxiv.org/pdf/0901.3725.pdf -
        The covariance matrix R describes the estimate of the error of the data; if
        the random errors in the entries of the data vector d are independent,R is
        diagonal and its diagonal entries are the squares of the standard
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
        self.results = [self.state_mean]

    def step(self, data=None):
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
        if not data is None:
            self.update(data)
            self.update_models()
        self.time += 1
        self.update_state_mean()
        self.results.append(self.state_mean)

    def predict(self):
        """
        Step the model forward by one time-step to produce a prediction.

        Params:

        Returns:
            None
        """
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
        for i in range(self.ensemble_size):
            diff = self.data_ensemble[:, i] - self.H @ self.state_ensemble[:, i]
            X[:, i] = self.state_ensemble[:, i] + gain_matrix @ diff
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
        diff = state_covariance - self.data_covariance
        return C @ self.H_transpose @ np.linalg.inv(diff)

    @staticmethod
    def is_good_model(model):
        """
        A utility function to ensure that we've been provided with a good model.
        This means that the model should have the following:
        - step (method)
        - state (attribute)
        - state2agent (method)
        - agent2state (method)
        
        Params:
            model

        Returns:
            boolean
        """
        methods = ['step', 'state2agent', 'agent2state']
        attribute = 'state'
        has_methods = [has_method(model, m) for m in methods]
        b = True if all(has_methods) and hasattr(mode, attribute) else False
        return b

    @staticmethod
    def has_method(model, method):
        """
        Check that a model has a given method.
        """
        b = True
        try:
            m = getattr(model, method)
            if not callable(m):
                w = "Model {} not callable".format(method)
                warns.warn(w, RuntimeWarning)
                b = False
        except:
            w = "Model doesn't have {}".format(method)
            warns.warn(w, RuntimeWarning)
            b = False
        return b
