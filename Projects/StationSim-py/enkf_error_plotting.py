"""
enkf_error_plotting.py
@author: ksuchak1990
date_created: 19/06/05
A script to create a bokeh visualisation to display variation in errors for the
enkf.
"""
# Imports
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models.widgets import Select
from bokeh.plotting import figure, output_file, show

# Read data
data = pd.read_csv('./results/enkf_300_20_20_20.csv')

# Set up static html output file
output_file('enkf_errors.html')

# Constants
SUBPLOT_WIDTH = 250
SUBPLOT_HEIGHT = 250
SCATTER_SIZE = 5
TIME = range(len(data['distance_errors']))

# Create subplots
subplot_dist = figure(width=SUBPLOT_WIDTH, height=SUBPLOT_HEIGHT, title='Distance')
subplot_dist.circle(TIME, data['distance_errors'], size=SCATTER_SIZE)

subplot_x = figure(width=SUBPLOT_WIDTH,
                   height=SUBPLOT_HEIGHT,
                   title='x',
                   x_range=subplot_dist.x_range,
                   y_range=subplot_dist.y_range)
subplot_x.circle(TIME, data['x_errors'], size=SCATTER_SIZE)

subplot_y = figure(width=SUBPLOT_WIDTH,
                   height=SUBPLOT_HEIGHT,
                   title='y',
                   x_range=subplot_dist.x_range,
                   y_range=subplot_dist.y_range)
subplot_y.circle(TIME, data['y_errors'], size=SCATTER_SIZE)

# Selection dropdowns
ap_select = Select(title='Assimilation phase',
                   value='20',
                   options=['2', '5', '10', '20', '50'])

es_select = Select(title='Ensemble size',
                   value='20',
                   options=['2', '5', '10', '20', '50'])

# Collect subplots with gridplot
p = gridplot([[ap_select, es_select, None],
             [subplot_dist, subplot_x, subplot_y]])
show(p)
