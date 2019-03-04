# Exercise File
'''
TODO:
	sspmm.py
	sspmm.md
	profile.py
	ParticleFilter.py
	ParticleFilter.md
'''
from data_assimilation.ParticleFilter import ParticleFilter
import numpy as np
import matplotlib.pyplot as plt
# from jupyterthemes.jtplot import style as jtstyle
# jtstyle('gruvbox')
import time


if 0:  # screensaver

	from models.screensaver import Model
	model = Model()
	pf = ParticleFilter(model, particles=10, window=1, do_copies=False, do_save=True)
	pf.batch(self, model, iterations=11, do_ani=False, agents=None)

if 1:  # sspmm

	from models.sspmm import Model
	model = Model({'pop_total': 20})

	if 0:  # Test Model
		model.batch()
	else:  # Test PF
		pf = ParticleFilter(model, particles=10, window=1, do_copies=True, do_save=False)
		pf.batch(model, iterations=10, do_ani=False, agents=None)
