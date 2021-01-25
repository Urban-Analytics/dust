"""
main.py
@author: ksuchak1990
Python script for running experiments with the enkf.
"""

# Imports
import numpy as np
from experiment_utils import Modeller, Visualiser, Processor

np.random.seed(42)

# Modeller.run_experiment_1(pop_size=100)
Processor.process_experiment_1()
