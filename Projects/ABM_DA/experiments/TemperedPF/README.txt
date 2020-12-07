Folder Contains stationsim_density models and particle filters for the AAMAS paper, both orginal and tempered versions.


particle_filter_AAMAS_temmper.py & stationsim_density_model_temper.py should be run together.
particle_filter_AAMAS.py & stationsim_density_model.py should be run together.


The number of tempers can be changed in line 297 of the particle_filter_AAMAS_temmper.py script by changing the upper limit of the dfactors range.


The step length of the Monte Carlo process that is initiated after each temper resample can be changed in line 253 of the stationsim_density_model_temper.py script.
To do this you must change the first number in the z variable calculation. This is automatically set to 70 pixels (5 meters).