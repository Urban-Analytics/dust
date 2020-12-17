#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 12:00:40 2020

@author: medrclaa

This file contains the functions for constructing an exit gfates probablity 
distribution for an agent given its current heading. Currently, this heading is
determined by the slope of the line through an agents start and current
locations. This could easily be expanded for more complicated agent paths
through an enormous number of methods. For example, taking the last say 5
observations of an agent and fitting a quadratic regression could
be used to predict exits of agents with parabolic paths.

"""

########
#imports
########

import numpy as np
import sys
from shapely.geometry import LineString, Point, Polygon, MultiLineString
import matplotlib.pyplot as plt
from descartes import PolygonPatch

default_colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()["color"]
sys.path.append("..")
sys.path.append("../..")
import modules.default_ukf_configs as configs
from modules.sensors import generate_Camera_Rect

sys.path.append("../../..")
sys.path.append("../../../..")
from stationsim.stationsim_model import Model


def start_gate_heading(start_position, position):
    """estimate which way the agent is heading using its entry and current positions
        
    Parameters
    ----------
    start_position, position : array_like
        `start position` starting position and current `position` of agents
        
    Returns
    -------
    angles : heading `angles` in radians anticlockwise about east.

    """

    #differenece between current and start. Displacement vector to calculate angle
    res = position - start_position 
    #split into x and y coordinates for arctan2
    x = res[0::2]
    y = res[1::2]
    #arctan2 is a special function for converting xy to polar. 
    #NOT THE SAME AS REGULAR ARCTAN
    angles = np.arctan2(y, x)
    return angles


def vision_polygon(position, angles, theta, boundary, dist = 1000):
    """Make a cone of vision for an agent theta degrees either side of its heading.
    
    - draw lines theta degrees either side of the heading about the current position
    - draw an arc segment using these two lines centred at the current position
    - cut any of the arc segment not in the stationsim boundary
    - return the cut arc segment
    - this can have 3-6 sides depending how many corners the agent sees
    
    Parameters
    ----------
    position : array_like
        numpy array (2nx1) of agent `position`s. Every 2 elements is an xy 
        coordinate.
    angles : array_like
        (nx1) array of `angles`. each element is the heading of an agent in radians
        anti-clockwise about east.
    theta : float
        angle `theta` for how wide the agents vision is. 0<theta<pi making the whole
        agentsvision angle 0 < 2theta < 2pi
    boundary : polygon
        `boundary` polygon of stationsim.
    dist : float, optional
        How large are the lines either side of the agents vision projected.
        These need to be much larger than the dimensions of the model otherwise
        the agent will not be able to see exit gates. The default is 1000.

    Returns
    -------
    polys : list
        `polys` list of polygons indicating what an agent can see. It is a cone
        of vision about its heading truncated by the stationsim boundary.

    """
    """If the agents field of vision is convex >180 degrees a different construction 
    is required. Drawing a polygon from the given list of coordinates
    will result in the complementary polygon of  angle <180 degrees being produced. 
    For example, we draw a polygon between three points we expect a triangle.
    However in this case we would want the entire rectangular boundary with 
    this triangle removed instead. Since we cannot draw this polygon without
    all of the boundary points included we  instaed draw the same triangle polygon
    but take the difference rather than the intersection with the boundary.
    !! a little plot would explain this better."""
    
    convex = False
    if theta >np.pi/2:
        convex = True

    #angles theta either side of the agent headings
    theta1 = (angles - theta) 
    theta2 = (angles + theta)
    
    polys = []
    for i in range(int(position.shape[0]/2)):
        p = position[(2*i):((2*i)+2)]
        #project lines theta either side of the headings
        line1 = LineString([Point(p), Point(p[0] + (dist * np.cos(theta1[i])),
                                       p[1] + dist * np.sin(theta1[i]))])
        line2 = LineString([Point(p), Point(p[0] + (dist * np.cos(theta2[i])),
                                       p[1] + (dist * np.sin(theta2[i])))])
        # Stack coords for one polygon. Note line2 is reversed so points are ordered properly.
        # This is an (4x2 array) of xy coords
        coords = np.vstack([np.array(line1.coords),
                           np.array(line2.coords[::-1]),
                           ])
        poly = Polygon(LineString(coords))
        #if the angle of vision is not convex we keep the normal polygon
        if not convex:
            cut_poly = boundary.intersection(poly)
        else:
            """ if the angle is convex >180 the polygon intersection will be
            for the complement shape with angle <180. Hence we want to take
            the complement of the complement shape to get back to where we started
            I.E. the convex angled polygon.
            """
            cut_poly = boundary.difference(poly)
        polys.append(cut_poly)        
    
    return polys

def cut_boundaries(polys, boundary):
    """take a subsection of  the boundary that each agent can see
    
    Parameters
    --------
    polys : list
        list of polygons `polys` of intersection between agents vision cone
        and boundary.
    boundary : polygon
         polygon for stationsim `boundary` 
         
    Returns
    -------
    cut_bounds : list
        `cut_bounds` list oflinestrings indicating which parts of the boundary
        the agent sees.

    """
    cut_bounds = []
    for poly in polys:
        cut_bounds.append(poly.exterior.intersection(boundary.exterior))
    return cut_bounds
    
def exit_gate_probabilities(gates, cut_boundaries, scale = True):
    """ Build a probability distribution on the length of each gate in an agents vision

    - calculate intersection between exit gate circles and boundary exterior
        to get the amount of the boundary each exit gate takes up.
    - calculate intersection between the agent vision and the exit
        gate boundaries to see how much of each gate the agent sees.
    - given the length of each gate the agent sees reweight these lengths
        by their sum to get the relative proportion of each gate an agent can see.
    - These proportions will sum to 1 and be used as an empirical probability 
        distribution. For example, if an agent see two exit gates, but 9x more
        length of gate 1, then it will assign probability 0.9 that the agent
        is heading to gate 1 and 0.1 probability it is heading to gate 2.
    Parameters
    ----------
    gates, cut_boundaries : list
     lists of exit `gates` polygons and `cut_boundaries` LineStrings

    scale : bool
        if `scale` convert the amount of each gate in the boundary into proportions
        that sum to 1. For example if precisely two gates of the same width are both 
        entirely in an agents field of vision. We assign them both 0.5 
        proportion respectively.
        
    Returns
    -------
    None.

    """
    #loop over each agents cone of vision via  its cut_boundary
    # work out how much of each exit gate is contained within the 
    
    main_intersect_lengths = []
    
    for boundary in cut_boundaries:
        intersects = []
        intersect_lengths = []
        for gate in gates:
            intersect = boundary.intersection(gate)
            intersects.append(intersect)
            intersect_length = intersect.length

            intersect_lengths.append(intersect_length)
            
        intersect_length_sum = sum(intersect_lengths)
        if scale and  intersect_length_sum!= 0:
            intersect_lengths = [item/intersect_length_sum for item in intersect_lengths]
        
        main_intersect_lengths.append(intersect_lengths)
        
    gate_probabilities = np.array(main_intersect_lengths)

            
    return gate_probabilities
        
    
def exit_points_to_polys(exit_gates, boundary, buffer):
    """convert numpy array of gate centroids to list of circle polygons
    
    Parameters
    --------
    exit_gates : array_like
        `exit_gates` numpy array containting the central point of each
        (semi-)circular exit gate.
    boundary : polygon
        `boundary` of the stationsim corridor.
    buffer : float
        `buffer` radius of the exit polygon circles. usually 1.
    
    Returns
    -------
    exit_polys : list
        `exit_polys` list of polygons representings the the exit gates

    """
    exit_polys = []    
    for i in range(exit_gates.shape[0]):
        exit_poly = Point(exit_gates[i,:]).buffer(buffer)
        exit_poly = exit_poly.intersection(boundary)
        exit_polys.append(exit_poly)
        
    return exit_polys

def plot_vision(cut_bounds, exit_polys, polys, boundary):
    """plot an polygons of agents vision and exit gates.
    
    Parameters
    --------
    cut_bounds : lst
        list of `cut_bounds` indicating which sections of exit gates are seen
        by an agent (if any).
    
    exit_polys, polys : list
        `exit_polys` lists of the circular exit gate polygons and the 
        agent vision polygons `polys`.
    Returns
    -------
    None.

    """
    width = boundary.exterior.bounds[2]
    height = boundary.exterior.bounds[3]
    
    f = plt.figure()
    ax = f.add_subplot(111)

    for item in exit_polys:
        item = np.array(item.exterior.coords.xy).T
        plt.plot(item[:, 0], item[:, 1], color = "k")

    for i, item in enumerate(polys):
        
        patch = PolygonPatch(item, alpha = 0.1, color = default_colour_cycle[int(i%10)])
        ax.add_patch(patch)
        item = np.array(item.exterior.coords.xy).T
        plt.plot(item[:,0], item[:,1])

        
    for item in cut_bounds:
        if type(item) == MultiLineString:
            for sub_item in item:
                sub_item = np.array(sub_item.coords.xy).T
                plt.plot(sub_item[:,0], sub_item[:, 1], color = "red")
        else:
            item = np.array(item.coords.xy).T
            plt.plot(item[:,0], item[:, 1], color = "red")
            
    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    plt.title = "Each Agents vision of exit gates."
    plt.show()

def heading_importance_function(position, start_position, theta, boundary, exit_polys):
    """calculate empirical probabilities based on start and current agent positions
    

    Returns
    -------
    None.

    """
    angles = start_gate_heading(start_position, position)
    polys = vision_polygon(position, angles, theta, boundary)
    cut_bounds = cut_boundaries(polys, boundary)
    gate_probabilities = exit_gate_probabilities(exit_polys, cut_bounds)
    
    return gate_probabilities, polys, cut_bounds

def  main(n, importance_function):
    """test function for agent desnsities to assume it works
    

    Returns
    -------
    None.

    """
    
    model_params = configs.model_params
    model_params["pop_total"] = n
    ukf_params = configs.ukf_params
    
    base_model = Model(**model_params)
    start_position = np.hstack([agent.loc_start for agent in base_model.agents])
    end_position = np.hstack([agent.loc_desire for agent in base_model.agents])
    
    width = model_params["width"]
    height = model_params["height"]
    boundary = generate_Camera_Rect(np.array([0, 0]), 
                                    np.array([0, height]),
                                    np.array([width, height]), 
                                    np.array([width, 0]))
    buffer = base_model.gates_space
    exit_gates =  base_model.gates_locations[-base_model.gates_out:]
    exit_polys = exit_points_to_polys(exit_gates, boundary, buffer)
    
    
    
    
    for _ in range(10):
        base_model.step()
    
    position = base_model.get_state(sensor = "location")
    theta = np.pi/12
    #start_position = np.array([50, 50])
    #position = np.array([25, 25])
    #exit_gates = np.array([[0,5], [5,0]])
    #exit_polys = exit_points_to_polys(exit_gates, boundary, buffer)
    gate_probabilities, polys, cut_bounds = importance_function(position, 
                                                                start_position, 
                                                                theta, 
                                                                boundary, 
                                                                exit_polys)
    print(gate_probabilities)
    plot_vision(cut_bounds, exit_polys, polys, boundary)

if __name__ == "__main__": 
    n = 1
    main(n, heading_importance_function)
   