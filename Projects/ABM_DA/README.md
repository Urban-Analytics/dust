# Agent-Based Modelling and Data Assimilation

Experiments in developing data assimilation algorithms for ABMs.

## Core code overview: `stationsim` directory

The [stationsim](./stationsim) directory contains the main code to run agent-based models and associated data assimilation algorithms.

Codes to actually run the models and various scenarios are in the [experiments](./experiments) directory.

### `stationsim_model.py`

The original version of the stationsim model. This is an Agent-Based Simulation model of people walking in a station.


### `stationsim_gcs_model.py`

In this version of the StationSim model, a new collision definition is used so that agents can collide from any direction (in the original version the movements of the agents were assumed to be from the left of the environment to the right).

In addition, a new station structure is available. To understand these changes, see the jupyter notebook: [StationSim - Grand Central Station version](./experiments/gcs_experiments/StationSim_GrandCentral_version.ipynb)

To run experiments with the new model, see [`gcs_experiments`](../experiments/gcs_experiments/gcs_experiments.ipynb).

You will need to have [jupyter](https://jupyter.org/) installed to do that.
