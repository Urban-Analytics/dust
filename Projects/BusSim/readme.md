This is an Agent-Based Simulation model of the bus route. 

Created by: Minh Kieu, University of Leeds

Version control: 

 v1.0: 5th Nov 2011
  -A single bus route  
  -Random passenger demand (uniformly distributed)  
  -No dataset used  
  -No state trasition capabilities (not DA-ready)
  
    v1.3 8th Nov 2018
    - Demand comes from an OD table instead of completely random
    - Bus stop also stores visited bus and headways
    - Export bus positions data with noise
    - Export bus positions ground truth data without noise

    v1.4 12th Nov 2018
    - Random speed (Normal distributed)
    - generate noisy states data and groundtruth
    - Compile the data and export to a pickle
    - Proper model parameters
    - Proper model outputs

    v1.5 16th Nov 2018
    - Realistic speed increase to desired speed
    - Bug fixings
    - State has [status, bus postion, bus speed,Occupancy]
    
    v1.6 3th Dec 2028
    - Make ODTable and ArrivalRate inputs rather than just random, so that we can do replications
    - Bug fixings
    - There is now no passenger board from the first stop (for simplification)
    - Separate do_plot, do_ani and do_reps
