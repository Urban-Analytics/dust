# README

This folder contains the codes used to run Rob's experiments on the StationSim Unscented Kalman Filter ukf.py.

To run the experiments, use the [`run_ukf.py`](./run_pf.py) file. That script expects a single integer commandline input which corresponds to a particular configuration (number of agents,proportion of agents observed). These are specified in the script itself. E.g. to run experiment 17:

```
	python run_ukf.py 17
```

The `arc_ukf.sh` file runs all the experiments as a batch job by repeatedly calling `run_pf.py` with different integer inputs.

The code that actually does the filtering is in the [`stationsim`](../../stationsim) folder ([`ukf.py`](../../stationsim/particle_filter.py) specifically). This code is for the Environment linux servers (all jobs run on the same machine), whereas "arc_ukf.py" is for Arc (it creates a 'task array' which allows each experiment to be run as a separate job).

## Results

The pickled class instances for each experiment will end up in the [`ukf_results`](./results) folder. The script to read all the results is [`arc_depickle_test.py`] which depickles the classes and plots various metrics. 
