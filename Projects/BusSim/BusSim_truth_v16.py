"""
This is a "realistic" model of bus simulation. This model will generate the "observed data" and "ground truth data" for our implementation of DA on a simplified version of it. This ABM model aims to be as realistic as possible, it has several features:
- Random bus speed (Normal distributed)
- Random passenger arrivals (Poisson distributed)
- Add further white noise to the outputs

Time Interval should be in seconds, Distance should be in meters

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
    


Written by: Minh Kieu, University of Leeds
"""

# Import
import numpy as np
import matplotlib.pyplot as plt


class Passenger:

    def __init__(self, origin=None, dest=None, start_time=0):
        # Parameters to be defined
        self.origin = origin
        self.dest = dest
        self.start_time = start_time
        # Fixed parameters
        self.time_waited_for_bus = 0.0
        self.total_time = 0.0
        self.status = 1  # 1 for waiting, 2 for onboard, 3 for alighted passengers
        return

class Bus:

    def __init__(self, bus_params):
        # Parameters to be defined
        for key, value in bus_params.items():
            setattr(self, key, value)
        # Fixed parameters
        self.visited = 0  # visited bus stop
        self.states = []  # noisy states: [status,position,velocity]
        self.groundtruth = []  # ground-truth location of the buses (no noise)
        self.trajectory = []  # store the location of each buses
        return

    def move(self):
        self.position += self.velocity * self.dt
        return

class BusStop:

    def __init__(self, position, busstopID, arrival_rate, passengers,activation):
        # Parameters to be defined
        self.busstopID = busstopID
        self.position = position
        self.arrival_rate = arrival_rate
        # Fixed parameters
        self.passengers = []
        self.departure_rate = []
        self.actual_headway = []  # store departure times of buses
        self.arrival_time = [0]  # store arrival time of buses
        self.visited = []  # store all visited buses
        self.activation = activation

    def passenger_arrival(self, dest, current_time):
        """Add a new passenger to the queue."""
        passenger = Passenger(origin=self.busstopID, dest=dest, start_time=current_time)
        self.passengers.append(passenger)

class Model:

    def __init__(self, params):
        # Parameters to be defined
        for key, value in params.items():
            setattr(self, key, value)
        # Fixed parameters
        self.GeoFence = self.dt * 14 + 5  # an area to tell if the bus reaches a bus stop
        self.StopList = np.arange(0, self.NumberOfStop * self.LengthBetweenStop, self.LengthBetweenStop)  # this must be a list
        # Total number of bus being dispatched from stop 0
        self.FleetSize = int(self.EndTime / self.Headway)
        # a Origin-Destination table of passenger movements
        #self.ODTable = ODTable
        np.fill_diagonal(self.ODTable, 0)  # the diagonal should be all zeros
        # no one should come to the last stop to board a bus!
        self.ArrivalRate[-1] = 0
        self.current_time = 0  # current time step of the simulation
        self.alighted_passengers = []
        self.busstops = []
        self.buses = []
        self.TrafficSpeed = np.random.uniform(self.TrafficSpeed_mean - self.TrafficSpeed_std, self.TrafficSpeed_mean + self.TrafficSpeed_std)
        # Initial Condition
        self.initialise_busstops()
        self.initialise_buses()
        return

    def step(self, random_order=False):  # This is the main step function to move the model forward
        self.current_time += self.dt
        # randomly change the traffic speed every 50 time step
        if np.mod(self.current_time, self.dt*5) == 0:
            self.TrafficSpeed = np.random.uniform(self.TrafficSpeed_mean - self.TrafficSpeed_std, self.TrafficSpeed_mean + self.TrafficSpeed_std)
        if np.mod(self.current_time, self.dt*100) == 0:
            self.ArrivalRate = np.random.uniform(self.minDemand*(1+self.IncreasedDemand * self.current_time/(self.dt*100)) / 60, self.maxDemand*(1+self.IncreasedDemand * self.current_time/(self.dt*100))/ 60, self.NumberOfStop)
            for busstop in self.busstops:
                busstop.arrival_rate = self.ArrivalRate[busstop.busstopID]
               
        # STEP 1: loop through each bus stop and let passenger to arrive
        for busstop in self.busstops:
            if busstop.activation <= self.current_time:
                r = np.random.uniform()
                # probability of an arrival within the next time step is F(x) = 1 - exp(-lambda)
                F = 1 - np.exp(-busstop.arrival_rate * self.dt)
                if r < F:
                    # there is an arrival
                    # generate the destination stop from the OD table
                    if busstop.busstopID < self.NumberOfStop - 2:
                        dest = int(np.random.choice(np.arange(busstop.busstopID + 2, self.NumberOfStop), p=self.ODTable[busstop.busstopID + 2:self.NumberOfStop, busstop.busstopID] / sum(self.ODTable[busstop.busstopID + 2:self.NumberOfStop, busstop.busstopID])))
                    else:
                        dest = self.NumberOfStop - 1
                    # let a passenger arrives
                    busstop.passenger_arrival(dest, self.current_time)

        # STEP 2: loop through each bus and let it moves or dwells
        for bus in self.buses:
           # CASE 1: INACTIVE BUS (not yet dispatched)
            if bus.status == 0:  # inactive bus (not yet dispatched)
                # check if it's time to dispatch yet?
                # if the bus is dispatched at the next time step
                if self.current_time >= (bus.dispatch_time - self.dt):
                    bus.status = 1  # change the status to moving bus
                    # check if there is any passengers who are waiting for the bus at the first stop (Stop 0)
                    #boarding_count = min(len(self.busstops[0].passengers), (bus.size - len(bus.passengers)))
                    #for p in self.busstops[0].passengers[1:boarding_count]:
                    #    p.time_waited_for_bus = self.current_time - p.start_time
                    # add boarded passengers to the bus
                    #bus.passengers.extend(self.busstops[0].passengers[1:boarding_count])
                    # remove boarded passengers from the bus stop
                    #self.busstops[0].passengers = self.busstops[0].passengers[boarding_count + 1:]
                    # print('bus no: ',bus.busID, ' start moving at time: ',self.current_time)
            # CASE 2: MOVING BUS
            if bus.status == 1:  # moving bus
                bus.velocity = min(self.TrafficSpeed, bus.velocity + bus.acceleration * self.dt)
                bus.move()
                if bus.position > self.NumberOfStop * self.LengthBetweenStop:
                    bus.status = 3
                    bus.velocity = 0
                    bus.passengers = []

                # if after moving, the bus enters a bus stop with passengers on it, then we move the status to dwelling
                def dns(x, y): return abs(x - y)
                if min(dns(self.StopList, bus.position)) <= self.GeoFence:  # reached a bus stop
                    Current_StopID = int(min(range(len(self.StopList)), key=lambda x: abs(self.StopList[x] - bus.position)))  # find the nearest bus stop

                    if Current_StopID != bus.visited:  # make sure that this is the first visit
                        bus.visited = Current_StopID
                        # check if there is any onboards passengers who wants to alight to that stop
                        out_passengers = [passenger for passenger in bus.passengers if passenger.dest == Current_StopID]
                        # check if there is any passengers who are waiting for the bus
                        boarding_count = min(len(self.busstops[Current_StopID].passengers), (bus.size - len(bus.passengers)))
                        # keep a list of visited bus
                        self.busstops[Current_StopID].visited.extend(
                            [bus.busID])  # store the visited bus ID
                        if len(bus.passengers) > 0:
                            self.busstops[Current_StopID].departure_rate.extend([len(out_passengers) / len(bus.passengers)])
                        # if there is no one boarding or alighting, the bus would just move on, so status is still 1. Otherwise the bus
                        # has to stop, which means status should be 2
                        # if there is any one boarding or alighting
                        if len(out_passengers) > 0 or boarding_count > 0:
                            bus.status = 2  # change the status of the bus to dwelling
                            bus.velocity = 0  # bus stopped
                            # print('Bus ',bus.busID, 'stopping at stop: ',Current_StopID,' at time:', self.current_time)

                            if len(out_passengers) > 0:
                                """Passengers that reached their destination leave the bus."""
                                for passenger in out_passengers:
                                    # passenger.end_time = cur_time
                                    passenger.total_time = self.current_time - passenger.start_time
                                    # add the passengers to the list of alighted passengers
                                    self.alighted_passengers.append(passenger)
                                    # remove the passenger from the bus
                                    bus.passengers.remove(passenger)
                                # print('Alighting: ',len(out_passengers))

                            if boarding_count > 0:
                                """Passengers at the bus stop hop into the bus."""
                                for p in self.busstops[Current_StopID].passengers[1:boarding_count]:
                                    p.time_waited_for_bus = self.current_time - p.start_time
                                # add boarded passengers to the bus
                                bus.passengers.extend(self.busstops[Current_StopID].passengers[1:boarding_count])
                                # remove boarded passengers from the bus stop
                                self.busstops[Current_StopID].passengers = self.busstops[Current_StopID].passengers[boarding_count + 1:]
                                # print('Boarding: ',boarding_count)
                                # the bus must only leave the stop when all the boarding and alighting is done
                                bus.leave_stop_time = self.current_time + len(out_passengers) * self.AlightTime + boarding_count * self.BoardTime + self.StoppingTime  # total time for dwelling
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

            if bus.velocity > 0:
                NoisySpeed = max(0, bus.velocity + np.random.normal(0, self.Noise_std / 2))
            else:
                NoisySpeed = 0
            bus.states.append([bus.status, bus.position + np.random.normal(0, self.Noise_std), NoisySpeed, len(bus.passengers)])  # Now generate a noisy GPS signal of the bus
            bus.groundtruth.append([bus.status, bus.position, bus.velocity, len(bus.passengers)])
            bus.trajectory.extend([bus.position])
        return

    def initialise_busstops(self):
        for busstopID in range(len(self.StopList)):
            position = self.StopList[busstopID]  # set up bus stop location
            # set up bus stop location
            arrival_rate = self.ArrivalRate[busstopID]
            passengers = []
            if busstopID >0:
                activation = busstopID * int(self.LengthBetweenStop/self.TrafficSpeed_mean) - 1.5*60
            else: activation= 10e10  #the first stop will never be activated: no one will come their the board a bus
            # define the bus stop object and add to the model
            busstop = BusStop(position, busstopID, arrival_rate, passengers,activation)
            self.add_busstops(busstop)
        return

    def initialise_buses(self):
        for busID in range(self.FleetSize):
            bus_params = {
                "dt": self.dt,
                "busID": busID,
                # all buses starts at the first stop,
                "position": -self.TrafficSpeed_mean * self.dt,
                "passengers": [],
                "size": 100,
                "acceleration": self.BusAcceleration / self.dt,
                "velocity": 0,  # staring speed is 0
                "leave_stop_time": 9999,  # this shouldn't matter but just in case
                "dispatch_time": busID * self.Headway,
                "status": 0  # 0 for inactive bus, 1 for moving bus, and 2 for dwelling bus, 3 for finished bus
            }
            # define the bus object and add to the model
            bus = Bus(bus_params)
            self.add_buses(bus)
            # print('Finished initialising the bus ',busID, ' at location: ', position, ' will dipatch at: ', dispatch_time)
        return

    def add_buses(self, bus):
        self.buses.append(bus)
        return

    def add_busstops(self, busstop):
        self.busstops.append(busstop)
        return


def run_model(model_params,do_ani,do_spacetime_plot,do_reps):
    '''
    Model runing and plotting
    '''
   
    model = Model(model_params)

    if do_ani or do_spacetime_plot:
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
            if do_ani:
                text0 = ['Traffic Speed:  %.2f m/s' % (model.TrafficSpeed)]
                plt.text(0, 0.025, "".join(text0))
                text2 = ['Alighted passengers:  %d' % (len(model.alighted_passengers))]
                plt.text(0, 0.02, "".join(text2))
                plt.text(0, 0.015, 'Number of waiting passengers')
                plt.plot(model.StopList, np.zeros_like(model.StopList), "|", markersize=20, alpha=0.3)
                plt.plot(model.StopList, np.zeros_like(model.StopList), alpha=0.3)
                for busstop in model.busstops:
                    # this is the value where you want the data to appear on the y-axis.
                    val = 0.
                    plt.plot(busstop.position, np.zeros_like(busstop.position) + val, 'ok', alpha=0.2)
                    plt.text(busstop.position, np.zeros_like(busstop.position) + val + 0.01, len(busstop.passengers))
                for bus in model.buses:
                    plt.plot(bus.position, np.zeros_like(bus.position) + val, '>', markersize=10)
                    text1 = ['BusID= %d, occupancy= %d, speed= %.2f m/s' % (bus.busID + 1, len(bus.passengers), bus.velocity)]
                    # text1 = ['BusID= ',str(bus.busID+1),', occupancy= ',str(len(bus.passengers))]
                    if bus.status != 0:
                        plt.text(0, -bus.busID * 0.003 - 0.01, "".join(text1))
                # plt.axis([-10, 31000, -0.1, 0.1])
                plt.axis('off')
                plt.pause(1 / 30)
                plt.clf()
            plt.show()
        if do_spacetime_plot:
            plt.figure(3, figsize=(16 / 2, 9 / 2))
            plt.clf()            
            x = np.array([bus.trajectory for bus in model.buses]).T        
            t = np.arange(0, model.EndTime, model.dt)
            x[x <= 0 ] = np.nan
            x[x >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan
            plt.plot(t, x, linewidth=.5)
            plt.ylabel('Distance (m)')
            plt.pause(1/30)
            plt.show()
            
        '''
        Now export the output data
        '''
        fixed_params = {
            "GeoFence": model.GeoFence,
            "StopList": model.StopList,
            "FleetSize": model.FleetSize,
            "ODTable": model.ODTable,
            "ArrivalRate": model.ArrivalRate
        }
    
        StateData = model.buses[0].states
        for b in range(1, model.FleetSize):
            StateData = np.hstack((StateData, model.buses[b].states))
        StateData[StateData < 0] = 0
    
        import pandas as pd
        GroundTruth = pd.DataFrame(model.buses[0].groundtruth)
        for b in range(1, model.FleetSize):
            GroundTruth = np.hstack((GroundTruth, model.buses[b].groundtruth))
        GroundTruth[GroundTruth < 0] = 0
    
        ArrivalRate = ()
        for b in range(len(model.busstops)):
            ArrivalRate += tuple([model.busstops[b].arrival_rate])
    
        ArrivalData = ()
        for b in range(len(model.busstops)):
            ArrivalData += tuple([model.busstops[b].arrival_time])
    
        DepartureRate = list([0])
        for b in range(1, len(model.busstops)):
            DepartureRate.extend([np.mean(model.busstops[b].departure_rate)])
            
        return model, model_params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, StateData, GroundTruth

    if do_reps: 
        RepHeadway = []
        
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()        
        
        for b in range(len(model.busstops)):
            RepHeadway += list([model.busstops[b].headway])
            
        return RepHeadway
    #return model_params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, StateData, GroundTruth

def plot(model_params, StateData, GroundTruth):
    # plotting the space-time diagram
    GroundTruth[GroundTruth == 0] = np.nan
    GroundTruth[GroundTruth > (model_params['NumberOfStop'] * model_params['LengthBetweenStop'])] = np.nan
    StateData[StateData == 0] = np.nan
    StateData[StateData > (model_params['NumberOfStop']*model_params['LengthBetweenStop'])] = np.nan
    plt.plot(StateData[:, range(1, np.size(GroundTruth, 1), 4)])
    plt.show()
    return

if __name__ == '__main__':
    
    NumberOfStop=20
    minDemand=0.5
    maxDemand=2
    #Initialise the ODTable and ArrivalRate
    ODTable = np.random.rand(NumberOfStop, NumberOfStop)
    ArrivalRate = np.random.uniform(minDemand / 60, maxDemand / 60, NumberOfStop)
    #Initialise the model parameters
    model_params = {
        "dt": 10,
        "NumberOfStop": 20,
        "LengthBetweenStop": 2000,
        "EndTime": 6000,
        "Headway": 5 * 60,
        "BurnIn": 1 * 60,
        "AlightTime": 1,
        "BoardTime": 3,
        "StoppingTime": 3,
        "BusAcceleration": 3,  # m/s
        "IncreasedDemand": 0.2, # 0.1 means that the arrival rate would be increased by 10% every 100 time step
        "TrafficSpeed_mean": 14,  # m/s
        "TrafficSpeed_std": 3,  # m/s
        "minDemand": 0.5,
        "maxDemand": 1,
        "Noise_std": 5,
        "ODTable": ODTable,
        "ArrivalRate":ArrivalRate        
    }

    #runing parameters    
    
    do_reps = False #whether we should export the distribution of headways
    do_spacetime_plot = True
    do_ani = False
    
    if do_ani or do_spacetime_plot:
        model, model_params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, StateData, GroundTruth = run_model(model_params,do_ani,do_spacetime_plot,do_reps)
        import pickle
        with open('BusSim_data.pkl', 'wb') as f:
            pickle.dump([model_params, fixed_params, ArrivalRate, ArrivalData, DepartureRate, StateData, GroundTruth], f)
    
    if do_reps:
        NumReps = 1000 
        Headway = run_model(model_params,do_ani,do_spacetime_plot,do_reps) 
        for r in range(NumReps):
            RepHeadway = run_model(model_params,do_ani,do_spacetime_plot,do_reps) 
            for b in range(NumberOfStop):
                Headway[b].extend(RepHeadway[b])    
        import pickle
        with open('BusSim_headway_1000reps.pkl', 'wb') as f:
            pickle.dump([Headway], f)
 