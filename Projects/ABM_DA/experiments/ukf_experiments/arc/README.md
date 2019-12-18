## ARC High Performance Computing

To perform experiments using the UKF on stationsim we employ the use of the ARC3 supercomputer.

(https://arc.leeds.ac.uk/systems/arc3/)

This allows for much faster parallel running of multiple experiments but requires files `arc.py` and `arc.sh`. 
The python script simply sets up and runs some experiment on stationsim using the ukf, some experiment module, and its associated parameters.

We also have bash script `.sh` which allows us to run multiple experiments at once. We create a list of all possible parameter combinations we wish to run in `arc.py` and pass this list onto `arc.sh` as a task array. Each number of the task array corresponds to an index of the list and some unique set of parameters we wish to test in `arc.py`. Once an experiment is run with the given parameters, we save the entire `ukf_ss` class instance in ukf results for extraction to some local terminal later.

## ARC3 Usage Guide

To use the above files in ARC we some require initial set up. In a linux bash terminal run the following with `<USERNAME>` replaced as necessary:

```
ssh <USERNAME>@arc3.leeds.ac.uk
git clone https://github.com/Urban-Analytics/dust/
cd /nobackup/<USERNAME>/dust/Projects/ABM_DA/experiments/ukf_experiments
module load python python-libs
virtualenv mypython
source mypython/bin/activate
```

This logs the user onto the arc3 system, clones this repository, and sets up a python3 virtual environment to run the experiments in. With this environment, we can pip in packages as necessary depending on the desired experiment. We have two sets of packages:

```
#for experiments 0 and 1:

pip install imageio
pip install ffmpeg
pip install seaborn

#for experiment 2:

pip install shapely
```

We are now ready to run experiments using arc. To do this we require some experiment module to run.
We use `ex1_input`, the module for experiment 1, as an example. Say we wish to run experiments for 5 and 10 agents populations, 0.5 and 1.0 proportion observed, and 20 repetitions for each pop and prop pair. We go into `arc.py` and change the inputs for `ex1_input` as follows.

```
nano arc.py #open text editor

default experiment parameters:

170     num_age = [10,20,30]# 5 to 50 by 5
171     props = [0.25,0.5,0.75,1] #.2 to 1 by .2
172     run_id = np.arange(0,30,1) #20 runs

new desired parameters:

170     num_age = [5,10] # 5 to 10 by 5
171     props = [0.5,1.0] #.5 to 1 by .5
172     run_id = np.arange(0,20,1) #20 runs
```

With our new parameters defined we now calculate the total number of experiments. This is simply multiplying the length of each parameter list together to get all unique combinations N = 2x2x20 = 80. We must update `arc_ukf.sh` with this number such that it runs every experiment and does not run blank experiments.

```
nano arc.sh #open text editor

#$ -t 1-3 #run tasks 1 through 3

becomes

#$ -t 1-80 # run tasks 1 through 80
```

Now everything is ready to run the experiment in arc. To do this we use the simple command `qsub`.

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
nano arc_ukf.sh.o<job_id>.<task_id> #display python console text

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

This downloads all files to local to be processed by `depickle.py`. 

## Building Your Own ARC3 Module.

If you want to build your own arc experiment module, I strongly suggest you first look at the `ukf_modules` readme for information on how to build an experiment module first. These modules are very similar with only a small number of additions. The main aim here is to take the default parameter dictionaries for stationsim and the ukf defined in `default_ukf_configs` and append them with parameters necessary to run your desired experiment. I am going to use experiment 1 as an example of how to build an arc module script on top of the existing experiment module.

First we define the function with the model_params and ukf_params as input

```
def ex1_input(model_params,ukf_params):
```

Next we build the list for the `arc.sh` task array. We build three seperate lists for each individual parameter unique to the ukf run. We have lists for population (num_age), proportion observed (props), and the run id (run_id) which indicates the repetition number given some pair of the previous parameters. We combine these sub lists together giving us our complete task array list.

```
num_age = [10,20,30]  # 10 to 30 agent population by 10
props = [0.25,0.5,0.75,1]  # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
run_id = np.arange(0,30,1)  # 30 runs

param_list = [(x, y,z) for x in num_age for y in props for z in run_id]
```

Using this list, we then select one item of this list, determined by `arc.sh`, giving us some set of three parameters we wish to run. These parameters are set to n, prop, and run_id respectively. Note we assign the latter further on.

```
n =  param_list[int(sys.argv[1])-1][0]
prop = param_list[int(sys.argv[1])-1][1]
```

We first assign the population size to model parameters and then assign a large number of required parameters to the ukf parameters using `omission_params`. For more on this function I direct the reader to the `ukf_modules` directory readme. We also assign the run_id, some file name to save the pickle under, and some boolean `do_pickle` to confirm if we pickle at all to the filter parameters. 

```
model_params['pop_total'] = n
    
omission_params(n, prop, model_params, ukf_params)
    
ukf_params["run_id"] = param_list[int(sys.argv[1])-1][2]
ukf_params["f_name"] = "ukf_results/ukf_agents_{}_prop_{}-{}".format(      
                    str(int(model_params['pop_total'])).zfill(3),
                    str(float(ukf_params['prop'])),
                    str(ukf_params["run_id"]).zfill(3))

ukf_params["do_pickle"] = True
```

With these complete dictionaries, we run `ukf2.py` as normal and pickle the result into `ukf_results`. If we do not wish to pickle the entire class, we can also specify some `ex_save` function to provide an alternative to pickling. This can be anything the user desires. For example, `ex0_save` saves a 3x1 numpy array of error metrics to greatly reduce the size of the data. This is by default none, and hence by default `arc.py` pickles the whole `ukf_ss` class.

## Depickle
