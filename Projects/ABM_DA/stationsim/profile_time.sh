#!/bin/bash
# Shell script to profile the time taken by a python script.
# Breaks down time by top 20 function calls.
# Usage:
#   ./profile_time.sh script_name
# where script_name is the name of the script that you want to profile.
# Requirements:
#   cProfile (included in Python 3)
#   pstats (included in Python 3)
python -m cProfile -o profiling_results $1 
python process_profile_stats.py 
rm profiling_results
