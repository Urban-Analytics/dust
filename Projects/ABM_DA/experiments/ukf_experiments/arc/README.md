## ARC High Performance Computing

To perform experiments using the UKF on stationsim we employ the use of the ARC4 supercomputer.

(https://arc.leeds.ac.uk/systems/arc4/)

This allows for much faster parallel running of multiple experiments but requires some initial set up.

This readme will introduce how we set up a conda environment in ARC4 as well as the two files `arc.py` and `arc.sh` used to define and run large scale ukf runs. The python script `arc.py` sets up and runs an individual experiment on stationsim using the ukf, some experiment module, and its associated parameters. We also have a bash script `arc.sh` which coordinates the running of multiple experiments at once using a task array.

(https://arc.leeds.ac.uk/using-the-systems/why-have-a-scheduler/advanced-sge-task-arrays/)

We then show how results from arc experiments can be transfered to a local terminal, an example of overall diagnostics from
`depickle.py`, and the requisites needed to build a new arc experiment module.

## ARC3 Setup Guide

NOTE: THIS IS DEPRECATED AND FOR REFERENCE ONLY. THIS NOW WORKS ONLY ON ARC4. SEE THE NEW SETUP BELOW.

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

## ARC4 Setup Guide

Using ARC4 over ARC3 is preferable mainly due to its ability to use conda environments. This section shows how to get onto ARC4 from some local bash terminal (linux/mac etc.).

We have an example initialisation in ARC4 below which we will break down here. 

- First, we log onto ARC4 using ssh. This is either done directly if on site at Leeds or via remote access if not.
- The personal user storage is fairly small and so we then create and move to our own /nobackup folder on ARC4 with significantly more (albeit not backed up) storage. 
- We clone our dust repository into nobackup and then move. Note we place the environment into a new folder in the users /nobackup as to save space in the users backed up (and very small) base directory.
- We then build a conda virtual environment. Note this is built outside of the git repo so if you wish to reuse the same conda environment and reclone the repo for new experiments you can. Just ignore this and the next step again in the future. 
- We then activate the environment and load any desired packages into conda either via conda install or in the original environment command.
- Finally we move to the arc subdirectory in which we perform our experiments.

```
Example initialisation. Note to use this yourself change all instances of medrclaa to your own Leeds username.

#log in to arc4 using ssh.
#note if not on site at Leeds we also need to log in via remote-access
#ssh medrclaa@remote-access.leeds.ac.uk
ssh medrclaa@arc4.leeds.ac.uk

#move to /nobackup
#if no directory in /nobackup create one with 
# mkdir /nobackup/medrclaa
cd /nobackup/medrclaa
#clone dust repo into nobackup
git clone https://github.com/Urban-Analytics/dust/

# load anaconda
module load python anaconda
#create virtual environment. Note this is technically outside the git clone and does not need to be run again if you wish. #You can keep this or rebuild it every time you rerun this experiment/ import a new git clone. Also note the -p argument
#to save the environment in /nobackup for space reasons.
#Also note we can automatically load packages by naming them at the end of this line
conda create -p /nobackup/medrclaa/ukf_py python=3 numpy matplotlib scipy shapely imageio seaborn
#activate the conda environment.
source activate /nobackup/medrclaa/ukf_py

#move to ukf experiments folder
cd /nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/arc

#we can then load in python packages as desired using conda e.g.
conda install shapely
#or pip at your own risk e.g. 
pip install imageio-ffmpeg
```

## Running an Example Experiment in ARC4

We are now ready to run experiments given some experiment module. We use the module for ukf experiment 1 (see `ukf_ex1.py` in modules), as an example. Say we wish to run experiments for 5 and 10 agent populations, 0.5 and 1.0 (50% and 100%) proportion observed, and 20 repetitions for each pop and prop pair. We go into `arc.py` and change the inputs for the `ex1_input` function as follows:

```
#open text editor, given we are in the `arc` folder.
nano arc.py 

#default experiment parameters:

170     num_age = [10,20,30]# 10 to 30 by 10
171     props = [0.25,0.5,0.75,1] #.25 to 1 by .25
172     run_id = np.arange(0,30,1) #30 runs

#new desired parameters:

170     num_age = [5,10] # 5 to 10 by 5
171     props = [0.5,1.0] #.5 to 1 by .5
172     run_id = np.arange(0,20,1) #20 runs
```

With our new parameters defined we now calculate the total number of experiments. This is simply multiplying the length of each parameter list (num_age, props, and run_id) together to get the number of unique experiment combinations. In this case we have N = 2x2x20 = 80 experiments and must update `arc.sh` with this number such that it exactly the right number of experiments. If this number is say 2 it will only run the first 2 experiments and ignore the remaining 78. Likewise, if we choose 82 runs we will have two dead runs that can throw errors unneccesarily.

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

This initiates the job which comes with several useful commands and outputs.

```
qstat - gives various progress diagnostics for all running jobs.
qdel <job_id> - cancel current job with given id.
```

We can also check the active progress or errors of any individual experiment using the .o (observed) and .e (errors) files generated for each experiment. For example, if we ran the all 80 experiments above and wish to check upon the first one (index starts at 1 here.) we do the following:

```
# call desired .o or .e file using format
# nano arc.sh.o<job_id>.<task_id> 
#e.g. say we have a run with id 1373131 and check the first experiment.
nano arc.sh.o.1373131.1

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

#for any errors that occur
#nano arc.sh.e<job_id>.<task_id>
#e.g.
nano arc.sh.e.1373131.1
OMP: Warning #181: GOMP_CPU_AFFINITY: ignored because KMP_AFFINITY has been def$

# For the record this error will always occur and generally means the experiment succeded. I don't know why it occurs.

```

This downloads all files to local to be processed by `depickle.py`. 

## Building Your Own ARC Module.

If you want to build your own arc experiment module, I strongly suggest you first look at the `ukf_modules` readme for information on how to build an experiment module first. These modules are very similar with only a small number of additions. The main aim here is to take the default parameter dictionaries for stationsim and the ukf defined in `default_ukf_configs` and append them with parameters necessary to run your desired experiment. I am going to use experiment 1 as an example of how to build an arc module script on top of the existing experiment module.

arc requisites are:
- some ukf experiment module e.g. `ukf_ex1.py`
- some list of experiments. each item in the list provides certain parameters to the experiment module 
        e.g. population proportion and run_id for for experiment 1
- 

We define the overall arc function with the model_params and ukf_params as inputs:

```
def ex1_input(model_params,ukf_params):
```

Next we build the lists for the `arc.sh` task array. We build three seperate lists for each individual parameter unique to the ukf run. We have lists for population (num_age), proportion observed (props), and the run id (run_id) to differentiate between repeats of some pair of the previous parameters. We combine these sub lists together giving us our complete task array list `param_list`.

```
num_age = [10,20,30]  # 10 to 30 agent population by 10
props = [0.25,0.5,0.75,1]  # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
run_id = np.arange(0,30,1)  # 30 runs

param_list = [(x, y,z) for x in num_age for y in props for z in run_id]
```

Using this list, we then select one item of this list, determined by `arc.sh`, giving us a single set of three parameters we wish to run. These chosen parameters are set to n, prop, and run_id respectively. Note we assign the latter further on.

```
n =  param_list[int(sys.argv[1])-1][0]
prop = param_list[int(sys.argv[1])-1][1]
ukf_params["run_id"] = param_list[int(sys.argv[1])-1][2]
```

Using these three parameters we can then define the remaining parameters needed for the arc module. The population `n` and proportion `prop` both go into the `omission_params` function from experiment 1. This function appends the `model_params` and `ukf_params` dictionaries given `n` and `prop`. We also use every parameter to produce a file name to which the pickle results are saved to.

```
model_params, ukf_params, base_model = omission_params(n, prop,
                                                   model_params, ukf_params)

ukf_params["run_id"] = run_id
ukf_params["file_name"] = "ukf_agents_{}_prop_{}-{}".format(
        str(n).zfill(3),
        str(prop),
        str(run_id).zfill(3)) + ".pkl"
```

With these complete dictionaries, we run `ukf2.py` as normal under experiment 1 and pickle the result into `ukf_results`. If we do not wish to pickle the entire class, we can also specify some `ex_save` function to provide an alternative to pickling. This can be anything the user desires. For example, `ex0_save` saves a 3x1 numpy array of error metrics to greatly reduce the size of the data. This is by default none, and hence by default `arc.py` pickles the whole `ukf_ss` class.


## Analysis

If the experiment runs successfully it will output 80 pickled UKF class-dict instances which we wish to analyse. I did this by copying the files back into linux and performing post-hoc analysis in spyder. To copy the files to local use the following scp command in a local linux terminal (NOT ARC).

```
scp <USERNAME>@leeds.ac.uk:source_in_arc/* destination_in_linux/.

e.g.
scp <USERNAME>@arc3.leeds.ac.uk:/nobackup/<USERNAME>/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
```
If we are accessing arc remotely we have an intermediate server to go through and so use proxy jump. The remote access server at Leeds struggles as is so don't scp directly it unless you want a complementary lecture from the arc team.

https://superuser.com/questions/276533/scp-files-via-intermediate-host
With the format:

scp -oProxyJump=user@remote-access.leeds.ac.uk
e.g.
```
scp -oProxyJump=medrclaa@remote-access.leeds.ac.uk medrclaa@arc4.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/results/agg* /Users/medrclaa/new_aggregate_results
```


## Depickle

!!todo explain how depickle module is built
