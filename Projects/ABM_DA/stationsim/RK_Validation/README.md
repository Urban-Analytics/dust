## Stationsim RK Validation


We wish to establish whether the outputs of two different versions of our agent based model (ABM) StationSim in python and c++ produce the same output. Since our model is by nature highly stochastic and non-linear it is difficult to do this without seeding each model.

We generate two populations of StationSim models without any seeding and test statistically if there is any evidence the two groups produce distinct results. We do this by analysing the collision_history_loc attribute for each stationsim instance telling us where any collisions between two agents occured in the model. We calculate the Ripley's K (RK) trajectories for each set of collisions giving two samples of RK curves. Using these two populations of curves, we fit a panel regression in R. and test if there is a different between the two groups using a panel regression and ANOVA.

We have the main notebook `stationsim_validation.ipynb` as well as python `station_validation.py` and R `RK_population_modelling.R` files.

From a vanilla python 3.7 and R conda environment we require packages;

astropy
matplotlib
numpy
pandas
scipy
seaborn

for R in python notebook
rpy2
simplegeneric
tzlocal

For full use of R file also install ggplot2 and docstring. Jupyter really doesnt like import these packages for some reason.

TODO: docker image?
