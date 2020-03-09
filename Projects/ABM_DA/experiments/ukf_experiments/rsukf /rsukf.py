#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: RC

Attempt at monte carlo sigma point kalman filter (SPKF) augment using rejection sampling/
approximate bayesian computation (ABC).

Dubbing it the RSUKF.

The aim here is to convert aggregate counts into a sample of agent positions for processing in an ensemble 
filtering technique such as the enkf/ukf.

General gist of the algorithm is as follows:

-Assume we have a run a stationsim model observed via a grid of counts. After two time steps we gain two measurements at
    time steps 1 and 2 respectively (y_1 and y_2).

- We only know which squares our agents are in and wish to estimate their GPS positions using approximate bayesian
    computation (ABC).

- To do this, we randomly generate a vector of agents positions normally distributed about x and P.
    We process this through some transition function f which also gives a normally distributed vector of positions.
    
    
- We then put this through a measurement function converting it into counts
  z_2 using some distance metric. If z_2 is within some tolerance of the true counts y_2 we accept it as
    belonging to an ensemble of potential agent positions.

- We repeat the previous two steps until a suitable ensemble population has been reached.

- We then use this ensemble to predict the current state of our agents through any potential sigma point filter
    (ukf,pf,enkf).

- When we get our measurement at time step 3. We can then step our ensemble again, measure the distance between
    z_3 and y_3, keep any suitable members and replenish the sample as necessary using futher ABC.

- We repeat the above 5 steps as necessary until StationSim finishes.
"""
import numpy as np

import sys
sys.path.append("../../../stationsim")
from stationsim_model import Model


"""list of functions needed and their purposes

L1 metric for stationsim for comparing y and z.
Rejection Sampler.
Sigma Point generator
rskf class
rskf_ss class
"""


def L1(v1, v2):

    """ Calculate L1 (Manhattan) distance between two vectors of counts.

    Parameters
    ------

    v1, v2 : list

        lists `v1` and `v2` of counts for how many agents are in each of a set of polygons.

    Returns
    ------
    dist : float
        `dist` L1 distance between v1 and v2
    """

    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.sum(np.abs(v1-v2))
    return dist


def rejection_sampler(v1, v2, distance_metric, tol, **metric_kwargs):

    """ Test if v1 and v2 are within some proximity of each other given an arbitrary distance metric.

    Parameters
    ------

    v1, v2 : list
            lists `v1` and `v2` of counts for how many agents are in each of a set of polygons.
            typically v2 is the true data and v1 is some randomly generated sample to compare.

    distance_metric : func
        metric of distance between two vectors. Typically L1 metric for vectors of counts.

    tol : float
        tolerance `tol` indicating whether to reject v1 given its proximity to v2.
        lower tol implies tighter restrictions and potentially more time generating a sample.

    metric_kwargs : kwargs
        `metric_kwargs` any kwargs for the distance metric

    Returns
    ------
    keep : bool
        whether to `keep` v1 or not.
    """

    dist = distance_metric(v1, v2, **metric_kwargs)
    if dist <= tol:
        accept = True
    else:
        accept = False
    return accept


def generate_agent(polygon):

    """ Randomly generate an agent within a polygon.

    This is just two uniform distributions for a square polygon.
    Could use something like rejection sampling or a convex hull for more complex polygons.

    Parameters
    ----------
    polygon : object
        shapely `polygon` object to generate a point in

    Returns
    -------

    point : array_like
        `point` randomly generated agent coordinate
    """
    xmin = np.min(polygon.exterior.coords.xy[0])
    ymin = np.min(polygon.exterior.coords.xy[1])

    xmax = np.max(polygon.exterior.coords.xy[0])
    ymax = np.max(polygon.exterior.coords.xy[1])

    x = np.random.uniform(xmin, xmax, 1)
    y = np.random.uniform(xmin, xmax, 1)
    
    point = np.array([x,y])
    
    return point

def generate_population(counts, polygons, ):
    
    """ randomly generate a population based on grid counts.
    
    TODO: improve random sampling via importance sampling/hmc/etc.
    
    """
    
    points = []
    
    for i, poly in enumerate(polygons):
        count = counts[i]
        while count != 0:
            points.append(generate_agent(poly))
            count -= 1
        
        
    
def RSSP(sigmas, polygons, z1, z2,  max_sample):

    """ Rejection Sampling Sigma Points (RSSP) generating function

    - take a given sample
    - process it forwards using some transition function f
    - observed propagated points and compare with true observations.
    - keep sigma points within a certain tolerance
    - generate new points to replenish sample.

    Parameters
    ------
    sigmas: array_like
        `sigmas` list of active sigma points

    polygons : list
        list of `polygons` on which we observe. usually a square grid.

    z1, z2: list
        list of counts at current and next time steps.

    max_sample : int
        `max_sample` how many sigmas needed. add more if lost due to rejection.
    """

class rsukf():

    def __init__(self):
        pass

    def backwards(self, sigmas, z, f, h, ):

        """ Given new measurements we propagate forwards sigma points as follows:

        - process any active sigma points forwards through transition function f
        - compare observations of processed points against truth using measurement function h
        - keep all sigma points within tolerance
        - replenish sample with new points as the


        Returns
        -------

        """
        pass

    
    def forwards(self):
        pass

class rsukf_ss():

    def __init__(self):
        pass



