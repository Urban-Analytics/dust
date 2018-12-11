"""
This function calibrates the BusSim model

Written by: Minh Kieu, University of Leeds
    Update v1.3
    - Calibration using CEM (KL-D)


Arrival data from the model not from the data

"""
# Import
import numpy as np
import pickle
from scipy.stats import ks_2samp, truncnorm
#import matplotlib.pyplot as plt
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

    def __init__(self, position, busstopID, arrival_rate, departure_rate, activation):
        self.busstopID = busstopID
        self.position = position
        self.actual_headway = []  # store departure times of buses
        self.arrival_time = [0]  # store arrival time of buses
        self.arrival_rate = arrival_rate
        self.departure_rate = departure_rate
        self.activation = activation


class Model:

    def __init__(self, params, fixed_params, ArrivalRate, DepartureRate, TrafficSpeed):
        self.params = (params, fixed_params, ArrivalRate, DepartureRate, TrafficSpeed)
        [setattr(self, key, value) for key, value in params.items()]
        [setattr(self, key, value) for key, value in fixed_params.items()]
        # Initial Condition
        self.TrafficSpeed = TrafficSpeed
        self.ArrivalRate = ArrivalRate
        self.DepartureRate = DepartureRate
        self.current_time = 0  # current time step of the simulation
        self.initialise_busstops()
        self.initialise_buses()
        # self.initialise_states()
        return

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
                        # Store the visited stop
                        bus.visited = Current_StopID
                        # passenger arrival rate
                        arrival_rate = self.busstops[Current_StopID].arrival_rate
                        # passenger departure rate
                        departure_rate = self.busstops[Current_StopID].departure_rate
                        # Now calculate the number of boarding and alighting
                        boarding_count = 0
                        # if the bus is the first bus to arrive at the bus stop
                        if self.busstops[Current_StopID].arrival_time == [0]:
                            if self.busstops[Current_StopID].activation <= self.current_time:
                                #boarding_count = min(np.random.poisson(arrival_rate * self.BurnIn), (bus.size - bus.occupancy))
                                boarding_count = min(np.random.poisson(arrival_rate * (self.current_time - self.busstops[Current_StopID].activation)), (bus.size - bus.occupancy))
                            alighting_count = int(bus.occupancy * departure_rate)
                        else:
                            timegap = self.current_time - self.busstops[Current_StopID].arrival_time[-1]
                            boarding_count = min(np.random.poisson(arrival_rate * timegap / 2), bus.size - bus.occupancy)
                            # print(boarding_count)
                            alighting_count = int(bus.occupancy * departure_rate)
                        # If there is at least 1 boarding or alighting passenger
                        if boarding_count > 0 or alighting_count > 0:  # there is at least 1 boarding or alighting passenger
                            # print(boarding_count)
                            # change the bus status to dwelling
                            bus.status = 2  # change the status of the bus to dwelling
                            bus.velocity = 0
                            #print('Bus ',bus.busID, 'stopping at stop: ',Current_StopID,' at time:', self.current_time)
                            bus.leave_stop_time = self.current_time + boarding_count * self.BoardTime + alighting_count * self.AlightTime + self.StoppingTime  # total time for dwelling
                            bus.occupancy = min(bus.occupancy - alighting_count + boarding_count, bus.size)

                        # store the headway and arrival times
                        if self.busstops[Current_StopID].arrival_time[-1] != 0:
                            #print([self.current_time - self.busstops[Current_StopID].arrival_time[-1]])
                            self.busstops[Current_StopID].actual_headway.extend([self.current_time - self.busstops[Current_StopID].arrival_time[-1]])  # store the headway to the previous bus
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
            activation = busstopID * int(self.LengthBetweenStop / self.TrafficSpeed_mean) - 2 * 60
            # define the bus stop object and add to the model
            busstop = BusStop(position, busstopID, arrival_rate, departure_rate, activation)
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


def unpickle_initiation():
    # load up a model from a Pickle
    with open('C:/Users/geomlk/Dropbox/Minh_UoL/DA/ABM/BusSim/BusSim_data.pkl', 'rb') as f:
        # with open('/Users/minhlkieu/Dropbox/Minh_UoL/DA/ABM/BusSim/BusSim_data.pkl', 'rb') as f:
        model_params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, StateData, GroundTruth = pickle.load(f)
    return model_params, fixed_params


def solution2model(model_params, fixed_params, solution):
    # apply a solution to a model
    ArrivalRate = solution[0:(model_params['NumberOfStop'])]
    DepartureRate = solution[model_params['NumberOfStop']:(2 * model_params['NumberOfStop'])]
    TrafficSpeed = solution[-1]
    # load model
    model = Model(model_params, fixed_params, ArrivalRate, DepartureRate, TrafficSpeed)
    return model


def run_model(model):
    '''
    Model runing and plotting
    '''
    RepHeadway = []

    for time_step in range(int(model.EndTime / model.dt)):
        model.step()
    for s in range(len(model.busstops)):
        RepHeadway += list([model.busstops[s].actual_headway])
    return RepHeadway


if __name__ == '__main__':  # Let's run the model
    '''
    Step 1: Model initiation and starting solution
    '''
    # load model parameters
    model_params, fixed_params = unpickle_initiation()

    CEM_params = {
        'MaxIteration': 100,  # maximum number of iteration
        'NumSol': 500,  # number of solutions being evaluated at each iteration
        'NumRep': 10,  # number of replication each time we run the model
        'Elite': 0.20  # pick 15% of best solution for the next iteration
        }

    with open('C:/Users/geomlk/Dropbox/Minh_UoL/DA/ABM/BusSim/BusSim_headway_1000reps.pkl', 'rb') as f:
        # with open('Users/minhlkieu/Dropbox/Minh_UoL/DA/ABM/BusSim/BusSim_headway_1000reps.pkl', 'rb') as f:
        Headway = pickle.load(f)

    # Starting solution
    mean_arr = np.random.uniform(model_params['minDemand'] / 60, model_params['maxDemand'] / 60, model_params['NumberOfStop'] - 2)
    std_arr = 0.05 * np.ones(model_params['NumberOfStop'] - 2)
    mean_dep = np.sort(np.random.uniform(0.05, 0.5, model_params['NumberOfStop'] - 3))
    std_dep = 0.1 * np.ones(model_params['NumberOfStop'] - 3)
    mean_traffic = np.random.uniform(model_params['TrafficSpeed_mean'] - model_params['TrafficSpeed_std'], model_params['TrafficSpeed_mean'] + model_params['TrafficSpeed_std'])
    std_traffic = 0.5
    '''
    Step 2: Main CEM loop for model calibration
    '''
    Sol_archived = []
    best_PI = 0
    PI_archived = []
    for ite in range(CEM_params['MaxIteration']):
        '''
        Step 2.1: Generate solution
        '''
        print('Iteration number ', ite)
        Sol_arr = np.zeros([CEM_params['NumSol'], 1])
        for p in range(len(mean_arr)):
            lower = 0
            mu = mean_arr[p]
            sigma = std_arr[p]
            upper = mu + 3 * sigma
            temp_arr = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=CEM_params['NumSol'])
            Sol_arr = np.append(Sol_arr, temp_arr[:, None], axis=1)
        Sol_arr = np.hstack([Sol_arr, np.zeros([CEM_params['NumSol'], 1])])

        Sol_dep = np.zeros([CEM_params['NumSol'], 2])
        for p in range(model_params['NumberOfStop'] - 3):
            lower = 0
            mu = mean_dep[p]
            sigma = std_dep[p]
            upper = mu + 3 * sigma
            temp_arr = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=CEM_params['NumSol'])
            Sol_dep = np.append(Sol_dep, temp_arr[:, None], axis=1)
        Sol_dep = np.hstack([Sol_dep, np.ones([CEM_params['NumSol'], 1])])

        lower = 0
        mu = mean_traffic
        sigma = std_traffic
        upper = mu + 3 * sigma
        Sol_traffic = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=CEM_params['NumSol'])

        Sol0 = np.hstack([Sol_arr, Sol_dep, Sol_traffic[:, None]])

        '''
        Step 2.2: Evaluate each solution in Sol0
        '''

        PIm = []
        for m in range(CEM_params['NumSol']):
            # build up model
            model = solution2model(model_params, fixed_params, Sol0[m])
            # run the model for NumRep times (NumRep>1)
            Headway_sol = run_model(model)
            for r in range(CEM_params['NumRep'] - 1):
                model = solution2model(model_params, fixed_params, Sol0[m])
                RepHeadway = run_model(model)
                for b in range(model_params['NumberOfStop']):
                    Headway_sol[b].extend(RepHeadway[b])
            # Now compare the collected solution with the observed value
            pvalue_tempt = []
            for b in range(1, model_params['NumberOfStop']):
                pvalue = ks_2samp(Headway_sol[b], Headway[0][b])[0]
                pvalue_tempt.extend([pvalue])
            PIm.extend([np.mean(pvalue_tempt)])
        Elist = np.argsort(PIm)[-(int(CEM_params['Elite'] * CEM_params['NumSol'])):]
        print('Current best solution: ', PIm[Elist[-1]])  # best solution

        '''
        Step 2.3: Generate new solutions for the next iteration
        '''
        mean_arr = np.mean(Sol_arr[Elist], axis=0)[1:model_params['NumberOfStop'] - 1]
        std_arr = np.std(Sol_arr[Elist], axis=0)[1:model_params['NumberOfStop'] - 1]
        mean_dep = np.mean(Sol_dep[Elist], axis=0)[2:model_params['NumberOfStop'] - 1]
        std_dep = np.std(Sol_dep[Elist], axis=0)[2:model_params['NumberOfStop'] - 1]
        mean_traffic = np.mean(Sol_traffic[Elist], axis=0)
        std_traffic = np.std(Sol_traffic[Elist], axis=0)

        '''
        Step 2.4: store the best solution and the current mean & std of the current generation of solutions
        '''
        if PIm[Elist[-1]] > best_PI:
            best_PI = PIm[Elist[-1]]
            best_mean = Sol0[Elist[-1]]
        sol_mean = np.concatenate([mean_arr, mean_dep, [mean_traffic]])
        sol_std = np.concatenate([std_arr, std_dep, [std_traffic]])
        Sol_archived.append(sol_mean)
        Sol_archived.append(sol_std)
        PI_archived.append(PIm[Elist[-1]])

    '''
    Step 3: Store the final parameter solution of BusSim
    '''
    with open('BusSim_model_calibration.pkl', 'wb') as f:
        pickle.dump([model_params, fixed_params, best_mean, Sol_archived, PI_archived], f)
