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

#Folders

Breakdown of each of the folders at this level.

<<<<<<< HEAD
arc - Running of multiple UKF experiments using Leeds' HPC ARC4 as well as summary plots for these results.
misc_plots - Various miscellanious plots for papers and visualisation.
pickles - Storage of local ukf results as pickle files.
plots - Where all plots are saved.
results - Storage of pickles for arc4 multiple model runs. Kept separate from pickles folder for local storage so only make summary plots with the desired results. 
sphinx - Sphinx files for conversion of arc readme into an html.
tests - Testing for UKF. (WIP)
=======
arc - Running of multiple UKF experiments using Leeds' HPC ARC4 as well as summary plots for these results.\
misc_plots - Various miscellanious plots for papers and visualisation.\
pickles - Storage of local ukf results as pickle files.\
plots - Where all plots are saved.\
results - Storage of pickles for arc4 multiple model runs. Kept separate from pickles folder for local storage so only make summary plots with the desired results. \
sphinx - Sphinx files for conversion of arc readme into an html.\
tests - Testing for UKF.\
>>>>>>> 3a359664a6c319114e5000fd6f67d792d92abf26
ukf_old - Various deprecated items.

#Experiments

The breakdown of the 4 experiments is as follows:

Experiment 0 - This experiment is for calibrating the UKF for stationsim. The aim is to see if the UKF is preferable to pure observation and prediction alone. Otherwise there would not be much point assimilating.

Experiment 1 - This experiment is designed to emulate a partial observed population of agents. Some agents are fully observed over time and some are unobserved, taking only their initial entry points. The point is to see if the UKF can still consistently predict observed agents but also if unobsered agents can be estimated better than pure prediction alone.  

Experiment 2 - Deprecated experiment. Can the UKF instead predict aggregate counts data. Typically only gaussian data can be used but its interesting to see if it works?

Experiment 3 - This experiment assumes we dont know where each agent is going. Can both the gaussian positions of agents and the categorical choice of exit gate be estimated. The reversible jump methodology is employed to do this.


