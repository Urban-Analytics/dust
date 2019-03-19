"""
A quick script to process the profiling stats.
@author: ksuchak1990
date_created: 19/03/13
last_edited: 19/03/13
"""
# Imports
import pstats

# Script
stats = pstats.Stats('profiling_results')
stats.sort_stats('tottime')
stats.print_stats(20)
