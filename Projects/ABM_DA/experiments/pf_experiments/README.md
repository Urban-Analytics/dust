# README

This folder contains the codes used to run Kevin's experimentss on the StationSim particle filter.

The code to run the experiments is in the [stationsim](../../stationsim) folder ([particle_filter.py](../../stationsim/particle_filter.py) specifically).

## pf_script-*.sh

These scripts run the particle filter. The python script is set up to take an integer command line argument which is an index to a list that determines the number of agents and particles to run in that experiment. 

The one for Arc creates a 'task array' which allows each experiment to be run as a separate job.

## Results

The results will end up in the [results](./results) folder. The script to read all the results is agent_particle_plot.py. The path in this code
needs to be set to the location of the results. 
