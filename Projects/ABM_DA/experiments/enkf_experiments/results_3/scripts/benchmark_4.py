# Benchmarking experiment 4
# No updating with filter
# Gates randomised for all agents

# Imports
import numpy as np
import sys
sys.path.append('../../../../stationsim/')
from ensemble_kalman_filter import EnsembleKalmanFilter
from ensemble_kalman_filter import AgentIncluder
from ensemble_kalman_filter import GateEstimator
from ensemble_kalman_filter import ExitRandomisation
from stationsim_gcs_model import Model
sys.path.append('../')
from experiment_utils import Modeller

# Reproducibility
np.random.seed(42)

# Data paths
data_dir = '../results/data/baseline/'
model_dir = '../results/models/baseline/'
fig_dir = '../results/figures/baseline/'

# Constants
pop_size = 20

# Run
Modeller.run_enkf_benchmark_filter(ensemble_size=20, pop_size=pop_size,
                                   exit_randomisation=ExitRandomisation.ALL_RANDOM)
