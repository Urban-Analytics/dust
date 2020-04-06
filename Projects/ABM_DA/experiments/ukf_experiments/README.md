## Unscented Kalman Filter

This folder contains the code used to run Rob's experiments on StationSim using the Unscented Kalman Filter. We have several sub folders which we describe here.

#Modules

Folder contains code to run individual experiments. We have code for experiments 0-2 with 3-4 being experimental/unused as well as several other files needed for these to run. We also code for plotting ukf results and other miscellanious ideas such as agent clustering.

#Arc

For running multiple ukf experiments at once using the ARC3 HPC at Leeds. We also have code to parse multiple experiments
into summary plots.

To run the experiments, use the [`run_ukf.py`](./run_pf.py) file. That script expects a single integer commandline input which corresponds to a particular configuration (number of agents,proportion of agents observed). These are specified in the script itself. E.g. to run experiment 17:

```
	python run_ukf.py 17
```

The `arc_ukf.sh` file runs all the experiments as a batch job by repeatedly calling `run_pf.py` with different integer inputs.

The code that actually does the filtering is in the [`stationsim`](../../stationsim) folder ([`ukf.py`](../../stationsim/particle_filter.py) specifically). This code is for the Environment linux servers (all jobs run on the same machine), whereas "arc_ukf.py" is for Arc (it creates a 'task array' which allows each experiment to be run as a separate job).

