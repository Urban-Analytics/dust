# README

This folder contains the codes used to run Kevin's experimentss on the StationSim particle filter.

The code to run the experiments is in the [stationsim](../../stationsim) folder.

## pf_script.sh

The script to get ARC working is pf_script.sh. It's the same as before except I've introduced the task array variable which makes
this large parameter sweep a lot easier to run. 

## Results

The results will end up in the [results](./results) folder. The script to read all the results is agent_particle_plot.py. The path in this code
needs to be set to the location of the results. 
