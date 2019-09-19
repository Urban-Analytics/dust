#!/bin/bash
# Example script: sample_script.sh
# 
# This script can be sunmitted to the HPC job submission system.
# To do so, the following commands need to be run first to configure the
# environment and then submit the job:
#
# module load python/3.6.0 python-libs/3.1.0
# qsub pf_script-arc.sh 
#
# Note that the python file that actually runs the particle filter 
# (called run_pf.py) expects an integer. This is an index into a combination
# of [agents,particles,noise] (the three parameters that need to be set).
# This script uses the job submission sustem to run a load of different 
# jobs, each giving a different number to the run_pf.py script. in this way
# a range of parameters can be tested.
#
# Set current working directory
#$ -cwd
# Use current environment variables and modules
#$ -V
# Request hours of runtime
#$ -l h_rt=48:00:00
# Email if a run aborts
#$ -m a
# Select memory
#$ -l h_vmem=5G # was 15 for big runs
# Choose cores
#$ -pe smp 5
# Tell computer this is an array job with tasks from 1 to N
# This number is determined by the length of the param_list list. 
#$ -t 1-132
#Run the executable pf.py
python3 ./run_pf.py $SGE_TASK_ID
