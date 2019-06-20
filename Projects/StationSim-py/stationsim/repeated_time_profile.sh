#! /bin/bash
# A script to run time profiling multiple times

for number in {1..10}
do
    python -m cProfile -o profiling_results Model.py 
    python process_profile_stats.py 
    rm profiling_results
done
