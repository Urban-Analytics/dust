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

## Usage Guide

To use the experiment files here in arc we require initial set up. In a linux bash terminal run the following with USERNAME replaced as necessary:

```
ssh USERNAME@arc3.leeds.ac.uk
git clone https://github.com/Urban-Analytics/dust/
cd /nobackup/USERNAME/dust/Projects/ABM_DA/experiments/ukf_experiments
module load python python-libs
virtualenv mypython
source mypython/bin/activate
```

This logs the user onto the arc3 system setting up a python3 virtual environment to run the experiments in. Now we can pip in the desired packages as necessary depending on the desired experiment. We have two sets of packages:

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

We now have a fully equipped environment for our experiments. We will now run an example experiment in which we run a basic experiment in arc_ukf.py. We will run the UKF 10 times for 5 and 10 agents at 0.5 and 1.0 proportion observed. First we define the parameters we wish to run in ['arc_ukf.py'].

```
nano arc_ukf.py #open text editor

default experiment parameters:

65    num_age = np.arange(5,55,5) # 5 to 50 by 5
66    props = np.arange(0.2,1.2,0.2) #.2 to 1 by .2
67    run_id = np.arange(0,20,1) #20 runs

new desired parameters:

65    num_age = [5,10] # 5 to 10 by 5
66    props = [0.5,1.0] #.5 to 1 by .5
67    run_id = np.arange(0,10,1) #10 runs
```

With our new parameters defined we calculate the total number of experiments. This is simply multiplying the length of each parameter list together N = 2x2x20 = 80. We must update ["arc_ukf.sh"] with this number such that it runs every experiment or does not run blank experiments.

```
nano arc_ukf.py #open text editor

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
