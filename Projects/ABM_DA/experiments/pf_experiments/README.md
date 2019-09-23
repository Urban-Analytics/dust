# README

This folder contains the codes used to run experiments with a particle filter. The particle filter code itself is available in the [stationsim](../../stationsim) package, called [particle_filter.py](../../stationsim/particle_filter.py). The work has been published in arXiv as a preprint prior to submission to a peer-reviewed journal:

 - Malleson, Nick, Kevin Minors, Le-Minh Kieu, Jonathan A. Ward, Andrew A. West, Alison Heppenstall
(2019) Simulating Crowds in Real Time with Agent-Based Modelling and a Particle Filter. _Preprint_:
[arXiv:1909.09397 [cs.MA]](https://arxiv.org/abs/1909.09397).

(Full source of the article available in [Writing/2019-ParticleFilter-Preprint/](../../Writing/2019-ParticleFilter-Preprint/)).


## Running Experiments

To run the experiments, use the [`run_pf.py`](./run_pf.py) file. That script expects a single integer command-line input which corresponds to a particular experiment configuration (number of agents, number of particles, amount of noise). These are specified in the script itself. E.g. to run experiment 17:

```
python run_pf.py 17
```

To run all the experiments in one go, there are two scripts that can be used:

 -  [`run_experiments-desktop.sh`](./run_experiments-desktop.sh) can be used to run the experiments on a normal desktop PC. It starts a for-loop that runs from 1 to N (where N is the total number of experiments) and simply calls `run_pf.py` each time round the loop, passing it an integer command line argument.

 -  ['run_experiments-grid.sh '](./run_experiments-grid.sh ) can be used on the University of Leeds [Advanced Research Computing (ARC)](https://arc.leeds.ac.uk/) high performance compute grid, or similar systems that use Sun Grid Engine (SGE). That script also uses `run_pf.py` to run the experiments, but it runs each experiment as a separate job so that all experiments can be run simultaneously across a grid.


## Analysing Results

The results will end up in the [`results`](./results) folder. The easiest way to analyse and visualise the results is using the iPython Jupyter Notebook: [`pf_experiments_plots.ipynb`](pf_experiments_plots.ipynb). You will need to have [jupyter](https://jupyter.org/) installed to do that.
