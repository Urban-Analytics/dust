# README

This folder contains the codes used to run Kevin's experimentss on the StationSim particle filter.

To run the experiments, use the [`run_pf.py`](./run_pf.py) file. That script expects a single integer commandline input which corresponds to a particular configuration (number of agents, number of particles, amount of noise). These are specified in the script itself. E.g. to run experiment 17:

```
	python run_pf.py 17
```

The `pf_script-*.sh` files run all the expeirments as a batch job by repeatedly calling `run_pf.py` with different integer inputs.

The code that actually does the filtering is in the [`stationsim`](../../stationsim) folder ([`particle_filter.py`](../../stationsim/particle_filter.py) specifically). One is for the Environment linux servers (all jobs run on the same machine), the other is for Arc (it creates a 'task array' which allows each experiment to be run as a separate job).

## Results

The results will end up in the [`results`](./results) folder. The script to read all the results is [`pf_experiments_plots.py`](pf_experiments_plots.py). 
