# README

This folder contains the codes used to run Kevin's experimentss on the StationSim particle filter. It has its own version of StationSim so doesn't depend on other libraries.

## StationSim-ARCExperiments.py

The code for the particle filter to run on ARC is StationSim-Experiments.py. It's all the standard stuff. The main difference is that the number of particles and agents is fed into the code from the ARC task array variable. This variable increments through 'param_list' and then
the particle and agent numbers are selected from there.

## pf_script.sh

The script to get ARC working is pf_script.sh. It's the same as before except I've introduced the task array variable which makes
this large parameter sweep a lot easier to run. 

## Results

The results should end up in a results folder. The script to read all the results is agent_particle_plot.py. The path in this code
needs to be set to the location of the results. 
