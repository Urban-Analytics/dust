#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:01:00 2020

@author: medrclaa
"""

import numpy as np
import shapely.geometry as geom

class camera_Sensor():
    
    
    def polygon_generator(pole, centre, arc, boundary):
        """construct Polygon object containing cone

        Parameters
        ------
        
        pole : array_like
            `pole` central point where the camera is. 
            The vision arc segment is centered about this point.
            
        centre : array_like
            `centre` indicates from the pole where the camera is facing.
            The distance of this point away from the pole also determines
            the radius of the cameras vision arc. Further away implies more 
            vision
            
        arc: float
            We choose a number 0 < `arc` <= 1 that determines how wide the vision 
            segment is. For example, if we choose arc = 0.25, the camera would form
            a quarter circle of vision centered about the line drawn
            between pole and centre.
               
        boundary : Polygon
             `boundary` of ABM topography. E.g. rectangular corridor for stationsim.

        Returns
        -------
        segment : Polygon
            `segment` polygon arc segment of cameras vision.

        """
        
        angle = arc * np.pi*2 # convertion proportion to radians
        diff = centre-pole # difference between centre and pole for angle and radius
        r = np.linalg.norm(diff) # arc radius
        centre_angle = np.arctan2(diff[1],diff[0]) # centre angle of arc (midpoint)
        
        #generate points for arc polygon
        precision = angle/10
        angle_range = np.arange(centre_angle - angle/2, centre_angle + angle/2 , precision)
        x_range = pole[0] +  r * np.cos(angle_range)
        y_range = pole[1] +  r * np.sin(angle_range)
    
        if arc < 1:
            # if camera isnt a complete circle add central point to polygon
            x_range = np.append(x_range, pole[0])
            y_range = np.append(y_range, pole[1])
  
        poly = geom.Polygon(np.column_stack([x_range,y_range]))
        poly = poly.intersection(boundary) #cut off out of bounds areas
        
        return poly
        
    
    def __init__(self, pole, centre, arc, boundary):
        self.pole = pole
        self.centre = centre
        self.arc = arc
        self.polygon = self.polygon_generator(centre, arc, boundary)
        
        
    def observe(self, agents):
        """ determine what this camera observes
        
        if an item if in the arc segment polygon take its observations (with noise?)
        Parameters
        ------
        agents: list
         list of `agent` classes 
        """
        
        
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
        self.agents = agents
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
        