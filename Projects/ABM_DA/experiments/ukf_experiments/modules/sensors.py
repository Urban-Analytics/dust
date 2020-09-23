#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A file containing all the imitation sensors used in applying the ukf
to stationsim
"""

import numpy as np
import shapely.geometry as geom

def generate_Camera_Rect(bl, tl, tr, br, boundary = None):
    """ Generate a square polygon 
    
    Parameters
    ------
    br, br, tl, tr : `array_like`
    bottom left `bl`, bottom right `br`, top left `tl`, and top right `tr`.
    Basically, where are the corners of the rectangle.
    
    boundary : `Polygon`
        indicate if this rectangle is a boundary. If no arguement is given,
        the polygon generated does is not an intersect with some boundary.
        This arguement is None when we generate a boundary and is usually
        stationsims dimensions if we wish to cut off some square cameras
        accordingly.
    Returns
    ------
    poly : `Polygon`
        square polygon with defined corners.
    """
    
    #build array of coordinates for shapely to read
    points = np.array([bl, tl, tr, br])
    #build polygon
    poly =  geom.Polygon(points)
    #take the intersect with some bounadary else return it as is
    if boundary is not None:
        poly = poly.intersection(boundary) #cut off out of bounds areas
        
    return poly

def generate_Camera_Cone(pole, centre, arc, boundary):
        """construct Polygon object containing cone

        I'd recommend knowing how polar coordinates work before reading this 
        code. Particualy converting cartesian to polar and back.
        Parameters
        ------
        
        pole : array_like
            `pole` central point where the camera is. 
            The vision arc segment originates about this point.
            
        centre : array_like
            `centre` indicates from the pole where the camera is facing.
            The distance of this point away from the pole also determines
            the radius of the cameras vision arc. Further away implies a 
            larger more radius and a long field of vision.
            
        arc: float
            We choose a number 0 < x <= 1 that determines how wide the vision 
            segment is. For example, if we choose arc = 0.25, the camera would form
            be a quarter circle of vision.
            
        boundary : Polygon
             `boundary` of ABM topography. E.g. rectangular corridor for 
             stationsim. Indicates where to cut off polygons if they're out of
             bounds.

        Returns
        -------
        poly : Polygon
            `poly` polygon arc segment of the cameras vision.
        """
        # convert arc from a proportion of a circle to radians
        angle = arc * np.pi*2 
        # difference between centre and pole.
        # determines which the radius and direction the camera points.
        diff = centre-pole 
        # arc radius
        r = np.linalg.norm(diff) 
        # angle the camera points in radians anticlockswise about east.
        centre_angle = np.arctan2(diff[1],diff[0]) 
        
        # generate points for arc polygon
        # precision is how many points. more points means higher resolution
        # set to 100 can easily be less or more
        precision = angle/100
        
        #start to build coordinates
        #start by finding out the angle of each polygon point about east

        angle_range = np.arange(centre_angle - angle/2,
                                centre_angle + angle/2 , precision)
        #convert these angles into x,y coordinates using the radius of the
        #camera r and the standard formula for polar to cartesian conversion
        x_range = pole[0] +  r * np.cos(angle_range)
        y_range = pole[1] +  r * np.sin(angle_range)
        
        if arc < 1:
            # if camera isnt a complete circle add central point to polygon
            # this essentially closes the loop making it a proper circle segment.
            x_range = np.append(x_range, pole[0])
            y_range = np.append(y_range, pole[1])
  
        #stack x and y coorindates and build polygon
        poly = geom.Polygon(np.column_stack([x_range,y_range]))
        #intersect with boundary to remove out of bounds elements.
        poly = poly.intersection(boundary) #cut off out of bounds areas
        
        return poly

class camera_Sensor():
    """class for an imitation camera."""
    
    def __init__(self, polygon):
        """init the camera using a single polygon
        !!maybe extend this to a list of polygons

        Parameters
        ----------
        polygons : `Polygon`
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.polygon = polygon
        
    def observe(self, state):
        """ determine what this camera observes of a model state at a given time
        
        if an item if in the arc segment polygon take its observations 
        !!with noise?
        
        Parameters
        ------
        agents: list
         list of `agent` classes 
        
        Returns
        ------
        which_in : list
        a list of the same shape a state indicating 1 if an agent is within a 
        cameras vision and 0 otherwise.
        """
        which_in = []
        
        state = np.reshape(state, (int(len(state)/2),2))
    
        
        for i in range(state.shape[0]):
            point = geom.Point(state[i,:])    
            
            is_in = point.within(self.polygon)
            if is_in:
                which_in += [i]
       
        return which_in
         
    
"WIP. Not yet working."
class footfall_Sensor():
    """
    count how many people pass through a specified polygon.
    """
    
    
    def __init__(self, polygon, positions):
        """
        
        Parameters
        ------
        polygon : Polygon
        """
        
        self.polygon = polygon
        self.agents = 0
        self.footfall_counts = []
        
    def count(self, agents):
        
        """ check if an agent posses through the footfall counter polygon
        
        draw lines between each new and old positions
        if lines intersect a polygon add one to count
        """
        footfall_count = 0
        # create list of Lines indicating where agents travelled
        old_agents = self.agents

        for i, item in enumerate(agents):
            line = np.array([[item.location, old_agents[i].location]])
            line = geom.LineString(line)              
            if line.intersects(self.polygon):
                footfall_count+=1
        self.footfall_counts.append(footfall_count)
        self.agents = agents
        