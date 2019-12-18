This is where all modules are stored pertaining to running experiments using the UKF on stationsim. The idea is to build an experiment module, such that the necessary prerequisites for running the ukf are met, but to be flexible enough such that any reasonable experiment can be run using one file `ukf2.py`.

## Constructing an Experiment Module

We will define the necessary prerequisites here and given an example module used for experiment 1 (random omission.). First a number of default parameters are defined for two dictionaries, named `model_params` and `ukf_params`, for stationsim model parameters and ukf parameters respectively. These contain a number of items

```
width - corridor width
height - corridor height
gates_in - how many entrances
gates_out - how many exits
gate_space- how wide are the exits 
gate_speed- mean entry speed for agents
speed_min - minimum agents speed to prevent ridiculuous iteration numbers
speed_mean - desired mean of normal distribution of speed of agents
speed_std - as above but standard deviation
speed_steps - how many levels of speed between min and max for each agent
separation - parameter to determine collisions in cKDTree. Similar to agent size
wiggle - how far an agent moves laterally to avoid queueing
step_limit - how many model steps to do before stopping.
do_ bools for saving plotting and animating data. 
```

```
sample_rate - how often to update kalman filter. higher number gives smoother predictions
do_batch - do batch processing on some pre-recorded truth data. not working yet
bring_noise - add noise to measurements?
noise - standard deviation of added Gaussian noise
a - alpha between 1 and 1e-4 typically determines spread of sigma points.
however for large dimensions may need to be even higher
b - beta set to 2 for gaussian. determines trust in prior distribution.
k - kappa usually 0 for state estimation and 3-dim(state) for parameters.
not 100% sure what kappa does. think its a bias parameter.
```

These parameters are required for every ukf experiment. They can be changed as necessary. For any module we also have further parameters that must be defined that can vary between experiments. For `model_params`:

```
pop_total - number of agents in abm
```

Then for `ukf_params`:

```
p - initial desired state covariance structure. Can be a guess or some kind of estimate. 
  Must be a numpy array with shape (2*pop_total,2*pop_total)
q - process noise. covariance structure for estimating covariance in ukf predict step.
  Must be a numpy array with shape (2*pop_total,2*pop_total)
r - sensor noise. covariance structure for updating covariance in update step.
  Must be a numpy array. Shape depends on function hx.
  
fx - transition function of Kalman Filter
hx - measurement function of Kalman Filter

obs_key_func - function that determines each agent is observed at each time step
  0 - unobserved, 1 - aggregated, 2 - gps style explicit observations
pickle_file_name - name to give pickle of `ukf_ss` instance for finished ABM run.
```

Finally in the main function we require

```
recall - Load an already complete pickle or run a new model from scratch.
do_pickle - do we save a newly ran model as a pickle.
pickle_source - where to save/load pickle files from
```

These parameters can be generated anyway the user desires. Typically they are generated from a small number of parameters. For example, in experiment 1 all of these parameters are generated from just the population size `n` and the proportion observed `prop`.
## Glossary
