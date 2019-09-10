# README

This folder contains the codes used to run Rob's experiments on the StationSim Unscented Kalman Filter ukf.py.

To run the experiments, use the [`run_ukf.py`](./run_pf.py) file. That script expects a single integer commandline input which corresponds to a particular configuration (number of agents,proportion of agents observed). These are specified in the script itself. E.g. to run experiment 17:

```
	python run_ukf.py 17
```

The `arc_ukf.sh` file runs all the experiments as a batch job by repeatedly calling `run_pf.py` with different integer inputs.

The code that actually does the filtering is in the [`stationsim`](../../stationsim) folder ([`ukf.py`](../../stationsim/particle_filter.py) specifically). This code is for the Environment linux servers (all jobs run on the same machine), whereas "arc_ukf.py" is for Arc (it creates a 'task array' which allows each experiment to be run as a separate job).

## Results

The pickled class instances for each experiment will end up in the [`ukf_results`](./results) folder. The script to read all the results is [`arc_depickle.py`] which depickles the classes and plots various metrics. 

## Usage Guide

To use the experiment files here in arc we require initial set up. In a linux bash terminal run the following with `<USERNAME>` replaced as necessary:

```
ssh <USERNAME>@arc3.leeds.ac.uk
git clone https://github.com/Urban-Analytics/dust/
cd /nobackup/<USERNAME>/dust/Projects/ABM_DA/experiments/ukf_experiments
module load python python-libs
virtualenv mypython
source mypython/bin/activate
```

This logs the user onto the arc3 system setting up a python3 virtual environment to run the experiments in. Now we can pip in packages as necessary depending on the desired experiment. We have two sets of packages:

```
for arc_base_config.py and arc_ukf.py

pip install imageio
pip install filterpy
pip install ffmpeg
pip install seaborn

for arc_ukf_agg.py also install:

pip install shapely
pip install geopandas
```

In this environment we run an example experiment for the basic ukf using `arc_ukf.py`. We will run the UKF 20 times for 5 and 10 agents at 0.5 and 1.0 proportion observed. First we define the parameters we wish to run in `arc_ukf.py`.

```
nano arc_ukf.py #open text editor

default experiment parameters:

62    num_age = [10,20,30]# 5 to 50 by 5
63    props = [0.25,0.5,0.75,1] #.2 to 1 by .2
64    run_id = np.arange(0,30,1) #20 runs

new desired parameters:

62    num_age = [5,10] # 5 to 10 by 5
63    props = [0.5,1.0] #.5 to 1 by .5
64    run_id = np.arange(0,20,1) #20 runs
```

With our new parameters defined we calculate the total number of experiments. This is simply multiplying the length of each parameter list together N = 2x2x20 = 80. We must update `arc_ukf.sh` with this number such that it runs every experiment and does not run blank experiments.

```
nano arc_ukf.sh #open text editor

#$ -t 1-3

becomes

#$ -t 1-80
```

Now everything is ready to run the experiment in arc. To do this we use the simple command qsub.

```
qsub arc_ukf.sh
```

This initiates the job and comes with several useful commands and outputs.

```
qstat - gives various progress diagnostics for any running job
qdel <job_id> - cancel current job
```

We can also check the active progress or errors of each job using text files generated in the current working directory

```
for ipython console
nano arc_ukf.sh.o<job_id>.<task_id>

## Hosts assigned to job 1373131.1:
##
## dc2s5b3d.arc3.leeds.ac.uk 5 slots
##
## Resources granted:
##
## h_vmem = 15G (per slot)
## h_rt   = 48:00:00
## disk   = 1G (per slot)

UKF params: {'Sensor_Noise': 1, 'Process_Noise': 1, 'sample_rate': 1, 'do_restr$
Model params: {'pop_total': 10, 'width': 200, 'height': 100, 'gates_in': 3, 'ga$
        Iteration: 0/3600
        Iteration: 100/3600
        Iteration: 200/3600
        Iteration: 300/3600
        Iteration: 400/3600
0:27:34.378509

for any errors that occur

nano arc_ukf.sh.e<job_id>.<task_id>

OMP: Warning #181: GOMP_CPU_AFFINITY: ignored because KMP_AFFINITY has been def$

```

## Analysis

If the experiment runs successfully it will output 80 pickled UKF class instances which we wish to analyse. I did this by copying the files back into linux and performing post-hoc analysis in spyder. To copy the files to local use the following scp command in a local linux terminal (NOT ARC).

```
scp <USERNAME>@leeds.ac.uk:source_in_arc/* destination_in_linux/.

e.g.
scp <USERNAME>@arc3.leeds.ac.uk:/nobackup/<USERNAME>/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
```

This downloads all files to local to be processed by `arc_depickle.py` and `grand_arc_depickle.py`. 

## Glossary

## arc_base_config.py and arc_base_config.sh

Used to provide precedent for various parameters in further experiments. Compares noisy observations, StationSim predictions, and UKF assimilations against true (noiseless) positions. This file simply runs a number of experiments and calculates the mean Average Euclidean Distance (AED) between each of the three estimates and the truth outputting a 3x1 numpy array per run. We vary assimilation rates (rates) and number of agents (num_age) 

The .sh script simply allows the .py file to be ran in arc. NOTE THE NUMBER OF TASKS N MUST BE ASSIGNED IN THE .sh SCRIPT `$# -t 1-N` ELSE NOT ALL EXPERIMENTS MAY BE RAN.

## arc_ukf.py and arc_ukf.sh

Basic Experiment running UKF on StationSim. The idea is to reduce the proportion of agents observed and see how the prediciton accuracy changes. We vary the number of agents (num_age) and proportion observed (prop).

This produces a pickled UKF class instance which we download to perform analysis on using `arc_depickle.py` and `grand_arc_depickle.py`

## arc_ukf_agg.py and arc_ukf_agg.sh

Similar to `arc_ukf` but with aggreated data rather than roughly known positions. This data is aggregated into various sized squares on which we test the efficacy of position prediction.  We vary the number of agents (num_age), square size (bin_size), and noise (noise).

## Depickles

Various scripts which depickle experiment output files into plots.

## base_config_depickle.py

Determines which of the three base config estimates perform best over varying observation noise and sampling rate.
Takes mean error over multiple runs for each of the observed,predcition and UKF metrics. Takes the minimum of the three means as the best performing metric for a given noise and sampling rate. Produces a chloropleth style map for the specified number of agents and lists of noises and rates.

## arc_depickle.py and grand_arc_depickle.py

The first produces more detailed diagnostics using multiple runs of a fixed number of agents for both `arc_ukf.py` and `arc_ukf_agg.py`. At each time point we sample the mean agent errors from each run as a population of means. The mean and variance of this sample are plotted to demonstrate the average error and uncertainty of the UKF over time. If the population is fully observed (as always with the aggregate case) then only one plot is produced. Otherwise both observed and unobserved plots are produced.

The `grand_arc_depickle.py` produces a more generalised diagnostic over multiple runs using multiple numbers of agents for `arc_ukf.py` only. This produces a chloropleth style map showing the grand mean error over both time and agents for various fixed numbers of agents and proportions observed.

NOTE: NEEDS AN EQUIVALENT FOR AGGREGATE CASE. SHOULD BE AS SIMPLE AS SWAPPING PROPORTION FOR BIN SIZE




