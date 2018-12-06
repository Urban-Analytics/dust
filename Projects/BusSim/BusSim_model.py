"""

This implements a bus simulation model as state transition, so compares to the original code it is DA-ready
The aim is to have the model as a state transition model, so that:

    x_{t+1} = Model (x_t)

We also simplify the model so that it runs more efficiently for real-time application
Time Interval should be in seconds, Distance should be in meters

This Real-time model:
    - Doesn't have Passenger Agents, but rather only model passengers through the number of boarding and alighting passengers
    - Should be more efficient than the Realistic bus model (needs profiling to double check!)
    - Doesn't know the Traffic Speed generated in the Realistic Model
    - Doesn't know how many passengers are awaiting at the next bus stop, but has some idea about the OD Table, Arrival Rate and Departure Rate

Here we try to solve a problem of missing info rather than noise in observed data
In real-time, BusSim doesn't know how many passengers are waiting and how many will alight at the downstream stops, and
there is also no data of the surrounding traffic

Can we use Data Assimilation to both update BusSim in real time, and also provide these missing information?

Written by: Minh Kieu, University of Leeds
    Update v1.3 5th Dec 2018
    - BusSim_model should use arrival_rate and departure_rate parameters from BusSim_model_calibration.py results (BusSim_model_calibration)
    - BusSim_model should use model_params and fixed_params
    

"""
# Import
import numpy as np
import matplotlib.pyplot as plt
from ParticleFilter import ParticleFilter
from copy import deepcopy
#import time

class Bus:

    def __init__(self, bus_params):
        # Parameters to be defined
        [setattr(self, key, value) for key, value in bus_params.items()]
        # Fixed parameters
        self.visited = 0  # visited bus stop
        self.trajectory = []  # store the location of each buses
        return
    def move(self):
        self.position += self.velocity * self.dt
        return


class BusStop:

    def __init__(self, position, busstopID, arrival_rate, departure_rate):
        self.busstopID = busstopID
        self.position = position
        self.actual_headway = []  # store departure times of buses
        self.arrival_time = [0]  # store arrival time of buses
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate
        #self.activation = activation

class Model:

    def __init__(self, params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, TrafficSpeed):
        self.params = (params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, TrafficSpeed)
        [setattr(self, key, value) for key, value in params.items()]
        [setattr(self, key, value) for key, value in fixed_params.items()]
        # Initial Condition
        self.TrafficSpeed = TrafficSpeed
        self.ArrivalRate = ArrivalRate
        self.ArrivalData = ArrivalData
        self.DepartureRate = DepartureRate
        self.current_time = 0  # current time step of the simulation
        self.current_step = 0
        self.initialise_busstops()
        self.initialise_buses()
        return

    def agents2state(self, do_measurement=True):
        '''
        This function stores the system state of all agents in a state vector with format:
            [bus_status1,bus_position1,bus_velocity1,...,bus_statusN,bus_positionN,bus_velocityN,busstop_arrivalrate1,busstop_departurerate1,...,busstop_arrivalrateM,busstop_departurerateM,TrafficSpeed]
        '''
        state_bus = np.ravel([(bus.status, bus.position, bus.velocity, bus.occupancy) for bus in self.buses])
        if do_measurement:
            state = state_bus  # other data not provided
        else:
            state_busstop = np.ravel([(busstop.arrival_rate, busstop.departure_rate) for busstop in self.busstops])
            # print(state_busstop)
            state_traffic = np.ravel(self.TrafficSpeed)
            state = np.concatenate((state_bus, state_busstop, state_traffic))
        return state

    def state2agents(self, state):
        '''
        This function converts the stored system state vectir back into each agent state
        '''
        # buses status, position and velocity
        for i in range(len(self.buses)):
            self.buses[i].status = int(state[4 * i])
            self.buses[i].position = state[4 * i + 1]
            self.buses[i].velocity = state[4 * i + 2]
            self.buses[i].occupancy = int(state[4 * i + 3])
        # bus stop arrival and departure rate
        for i in range(len(self.busstops)):
            self.busstops[i].arrival_rate = state[4 * self.FleetSize + 2 * i]
            self.busstops[i].departure_rate = state[4 * self.FleetSize + 2 * i + 1]
        # traffic speed
        self.TrafficState = state[-1]
        return
    
    def mask(self):
        return 1, None

    def step(self):
        '''
        This function moves the whole state one time step ahead
        '''
        # This is the main step function to move the model forward
        self.current_time += self.dt
        # Loop through each bus and let it moves or dwells
        for bus in self.buses:
            #print('now looking at bus ',bus.busID)
            # CASE 1: INACTIVE BUS (not yet dispatched)
            if bus.status == 0:  # inactive bus (not yet dispatched)
                # check if it's time to dispatch yet?
                # if the bus is dispatched at the next time step
                if self.current_time >= (bus.dispatch_time - self.dt):
                    bus.status = 1  # change the status to moving bus
                    bus.velocity = min(self.TrafficSpeed, bus.velocity + bus.acceleration * self.dt)

            if bus.status == 1:  # moving bus
                bus.velocity = min(self.TrafficSpeed, bus.velocity + bus.acceleration * self.dt)
                bus.move()
                if bus.position > self.NumberOfStop * self.LengthBetweenStop:
                    bus.status = 3  # this is to stop bus after they reach the last stop
                    bus.velocity = 0
                # if after moving, the bus enters a bus stop with passengers on it, then we move the status to dwelling

                def dns(x, y): return abs(x - y)
                if min(dns(self.StopList, bus.position)) <= self.GeoFence:  # reached a bus stop
                    # Investigate the info from the current stop
                    Current_StopID = int(min(range(len(self.StopList)), key=lambda x: abs(self.StopList[x] - bus.position)))  # find the nearest bus stop
                    
                    if Current_StopID != bus.visited:
                        #Store the visited stop                         
                        bus.visited = Current_StopID                                            
                        # passenger arrival rate
                        arrival_rate = self.busstops[Current_StopID].arrival_rate
                        print('Arrival rate: ',arrival_rate)
                        # passenger departure rate
                        departure_rate = self.busstops[Current_StopID].departure_rate
                        print('Departure rate: ',departure_rate)
    
                        # Now calculate the number of boarding and alighting
                        boarding_count = 0
                        # if the bus is the first bus to arrive at the bus stop
                        if self.busstops[Current_StopID].arrival_time == 0:
                            if self.busstops[Current_StopID].activation <= self.current_time:
                                #boarding_count = min(np.random.poisson(arrival_rate * self.BurnIn), (bus.size - bus.occupancy))
                                boarding_count = min(np.random.poisson(arrival_rate * (self.current_time-self.busstops[Current_StopID].activation)), (bus.size - bus.occupancy))
                            alighting_count = int(bus.occupancy * departure_rate)
                        else:
                            timegap = self.current_time - self.busstops[Current_StopID].arrival_time[-1]
                            boarding_count = min(np.random.poisson(arrival_rate*timegap/2), bus.size - bus.occupancy)                        
                            #print(boarding_count)
                            alighting_count = int(bus.occupancy * departure_rate)
                        # If there is at least 1 boarding or alighting passenger
                        if boarding_count > 0 or alighting_count > 0:  # there is at least 1 boarding or alighting passenger
                            # change the bus status to dwelling
                            bus.status = 2  # change the status of the bus to dwelling
                            bus.velocity = 0                            
                            bus.leave_stop_time = self.current_time + boarding_count * self.BoardTime + alighting_count * self.AlightTime + self.StoppingTime  # total time for dwelling
                            bus.occupancy = min(bus.occupancy - alighting_count + boarding_count, bus.size)
                        # store the headway and arrival times
                        if self.busstops[Current_StopID].arrival_time[-1] != 0:
                            #print([self.current_time - self.busstops[Current_StopID].arrival_time[-1]])
                            self.busstops[Current_StopID].actual_headway.extend([self.current_time - self.busstops[Current_StopID].arrival_time[-1]])# store the headway to the previous bus                            
                        self.busstops[Current_StopID].arrival_time.extend([self.current_time])  # store the arrival time of the bus

            # CASE 3: DWELLING BUS (waiting for people to finish boarding and alighting)
            if bus.status == 2:
                # check if people has finished boarding/alighting or not?
                # if the bus hasn't left and can leave at the next time step
                if self.current_time >= (bus.leave_stop_time - self.dt):
                    bus.status = 1  # change the status to moving bus
                    bus.velocity = min(self.TrafficSpeed, bus.velocity + bus.acceleration * self.dt)
            bus.trajectory.extend([bus.position])
        return

    def initialise_busstops(self):
        self.busstops = []
        for busstopID in range(len(self.StopList)):
            position = self.StopList[busstopID]  # set up bus stop location
            # set up bus stop location
            arrival_rate = self.ArrivalRate[busstopID]
            # set up bus stop location
            departure_rate = self.DepartureRate[busstopID]
            activation = busstopID * int(self.LengthBetweenStop/self.TrafficSpeed_mean) - 2*60
            # define the bus stop object and add to the model
            busstop = BusStop(position, busstopID, arrival_rate, departure_rate,activation)
            self.busstops.append(busstop)
        return

    def initialise_buses(self):
        self.buses = []
        for busID in range(self.FleetSize):
            bus_params = {
                "dt": self.dt,
                "busID": busID,
                # all buses starts at the first stop,
                "position": -self.TrafficSpeed_mean * self.dt,
                "occupancy": 0,
                "size": 100,
                "acceleration": self.BusAcceleration / self.dt,
                "velocity": 0,  # staring speed is 0
                "leave_stop_time": 9999,  # this shouldn't matter but just in case
                "dispatch_time": busID * self.Headway,                
                "status": 0  # 0 for inactive bus, 1 for moving bus, and 2 for dwelling bus, 3 for finished bus
                }
            # define the bus object and add to the model
            bus = Bus(bus_params)
            self.buses.append(bus)
        return


def unpickle_Model():

    import pickle

    if __name__ == '__main__':
        with open('BusSim_data.pkl', 'rb') as f: #load data from BusSim_truth
            model_params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, StateData, GroundTruth = pickle.load(f)
            
        with open('BusSim_model_calibration.pkl', 'rb') as f: #load data from BusSim_model_calibration
            model_params, fixed_params,best_mean,Sol_archived,PI_archived = pickle.load(f)
        
        ArrivalRate = np.concatenate([[0],best_mean[0:(model_params['NumberOfStop']-2)],[0]])
        DepartureRate = np.concatenate([[0,0],best_mean[(model_params['NumberOfStop']-2):-1],[1]])
        TrafficSpeed = best_mean[-1]
                
    model = Model(model_params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, TrafficSpeed)

    return model, GroundTruth, ArrivalRate_std, DepartureRate_std, TrafficSpeed_std

do_PF = True
#if __name__ == '__main__':  # Let's run the model
if do_PF == True:
    '''
    Step 2: Model runing and plotting
    '''
    model0, GroundTruth, ArrivalRate_std, DepartureRate_std, TrafficSpeed_std = unpickle_Model()
    t = np.arange(0, 10*len(GroundTruth), 10)

    for p in (100,):
        for std in (.1,):
            for w in (1,):
                print(p, std, w)
                name = '../images/BusSim/' + 'p{}std{}w{}'.format(p, std, w)
                formatting = '.pdf'

                filter_params = {
                    'number_of_particles': p,
                    'particle_std': std,
                    'resample_window': w,
                    'do_copies': True,
                    'do_save': True
                    }
                model = deepcopy(model0)
                pf = ParticleFilter(model, **filter_params)

                for niter in range(len(GroundTruth)):
                    model.step()
                    if 0:  # Use Model
                        true_state = model.agents2state()
                        measured_state = true_state + np.random.normal(0, 0., true_state.shape)
                    else:  # Use Truth
                        true_state = GroundTruth[niter]
                        measured_state = true_state + np.random.normal(0, 0., true_state.shape)
                    pf.step(measured_state, true_state)

                    do_ani = False
                    if niter == len(GroundTruth)-1 or do_ani:
                        plt.figure(3, figsize=(16 / 2, 9 / 2))
                        plt.clf()
                        for particle in range(len(pf.models)-1):
                            x = np.array([bus.trajectory for bus in pf.models[particle].buses]).T
                            x[x <= 0] = np.nan
                            plt.plot(t, x, linewidth=.5)
                        x = GroundTruth[:niter+1, 1::4]
                        x[x <= 0] = np.nan
                        plt.plot(t, x, 'k')
                        plt.xlabel('Time (s)')
                        plt.ylabel('Distance (m)')
                        if do_ani:
                            plt.pause(1/30)

                    plt.show()


    
