This will contain all deprecated files no longer used for various reasons as well as a glossary of what they do. 

## Glossary

## arc_base_config.py and arc_base_config.sh

Used to provide precedent for various parameters in further experiments. Compares noisy observations, StationSim predictions, and UKF assimilations against true (noiseless) positions. This file simply runs a number of experiments and calculates the mean Average Euclidean Distance (AED) between each of the three estimates and the truth outputting a 3x1 numpy array per run. We vary assimilation rates (rates) and number of agents (num_age) 

The .sh script simply allows the .py file to be ran in arc. NOTE THE NUMBER OF TASKS N MUST BE ASSIGNED IN THE .sh SCRIPT `$# -t 1-N` ELSE NOT ALL EXPERIMENTS MAY BE RAN.

## arc_ukf.py and arc_ukf.sh

Basic Experiment running UKF on StationSim. The idea is to reduce the proportion of agents observed and see how the prediciton accuracy changes. We vary the number of agents (num_age) and proportion observed (prop).

This produces a pickled UKF class instance which we download to perform analysis on using `arc_depickle.py` and `grand_arc_depickle.py`

## arc_ukf_agg.py and arc_ukf_agg.sh

Similar to `arc_ukf` but with aggreated data rather than roughly known positions. This data is aggregated into various sized squares on which we test the efficacy of position prediction.  We vary the number of agents (num_age), square size (bin_size), and noise (noise).

## Depickles

Various scripts which depickle experiment output files into plots.

## base_config_depickle.py

Determines which of the three base config estimates perform best over varying observation noise and sampling rate.
Takes mean error over multiple runs for each of the observed,predcition and UKF metrics. Takes the minimum of the three means as the best performing metric for a given noise and sampling rate. Produces a chloropleth style map for the specified number of agents and lists of noises and rates.

## arc_depickle.py and grand_arc_depickle.py

The first produces more detailed diagnostics using multiple runs of a fixed number of agents for both `arc_ukf.py` and `arc_ukf_agg.py`. At each time point we sample the mean agent errors from each run as a population of means. The mean and variance of this sample are plotted to demonstrate the average error and uncertainty of the UKF over time. If the population is fully observed (as always with the aggregate case) then only one plot is produced. Otherwise both observed and unobserved plots are produced.

The `grand_arc_depickle.py` produces a more generalised diagnostic over multiple runs using multiple numbers of agents for `arc_ukf.py` only. This produces a chloropleth style map showing the grand mean error over both time and agents for various fixed numbers of agents and proportions observed.

NOTE: NEEDS AN EQUIVALENT FOR AGGREGATE CASE. SHOULD BE AS SIMPLE AS SWAPPING PROPORTION FOR BIN SIZE
