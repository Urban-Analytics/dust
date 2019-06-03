
# Example script: sample_script.sh
#!/bin/bash
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
# Tell computer this is an array job with tasks from 1 to 3131.
# This number is determined by the length of the param_list list. 
#$ -t 1-3131
#Run the executable pf.py from the current working directory
python3 StationSim-ARCExperiments.py $SGE_TASK_ID
