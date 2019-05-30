"""
Filter.py
@author ksuchak1990
date_created: 19/05/30
A general base class on which to base filters
"""
# Imports
import warnings as warns

# Classes
class Filter:
    """
    An abstract class to represent a general filter.
    """
    def __init__(self, model, model_params):
        """
        Initialise the filter.

        Params:
            model

        Returns:
            None
        """
        # Instantiate the base model
        self.base_model = model(model_params)
        self.time = 0

        # Ensure that the model has the correct attributes
        assert self.is_good_model(self.base_model), 'Model missing attributes.'

    def step(self):
        """
        A placeholder method to step the filter.
        """
        pass

    @classmethod
    def is_good_model(cls, model):
        """
        A utility function to ensure that we've been provided with a good model.
        This means that the model should have the following:
        - step (method)
        - state (attribute)
        - set_state (method)
        - get_state (method)

        Params:
            model

        Returns:
            boolean
        """
        methods = ['step', 'set_state', 'get_state']
        attributes = ['state']
        has_methods = [cls.has_method(model, m) for m in methods]
        has_attributes = [hasattr(model, a) for a in attributes]
        b = all(has_methods) and all(has_attributes)
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
                w = "Model {0} not callable.".format(method)
                warns.warn(w, RuntimeWarning)
                b = False
        except:
            w = "Model doesn't have {0}.".format(method)
            warns.warn(w, RuntimeWarning)
            b = False
        return b
