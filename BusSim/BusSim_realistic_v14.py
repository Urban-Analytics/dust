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


Written by: Minh Kieu, University of Leeds
"""
# Import
import numpy as np
import matplotlib.pyplot as plt
         
class Passenger:
    def __init__(self,origin=None, dest=None, start_time=0):        
        #self.passengerID = passengerID        
        self.origin = origin
        self.dest = dest
        self.time_waited_for_bus = 0.0
        self.total_time = 0.0        
        self.start_time = start_time
        self.status = 1 # 1 for waiting, 2 for onboard, 3 for alighted passengers
        return

class Bus:        
    def __init__(self,dt,busID,position,passengers,size,total_passengers,velocity,leave_stop_time,dispatch_time,status):        
        self.dt = dt
        self.busID = busID        
        self.position = position
        self.visited = 0 #visited bus stop
        self.states = [] #noisy states: [status,position,velocity]
        self.groundtruth = [] #ground-truth location of the buses (no noise)
        #self.GPS = [] #GPS location of the buses (no noise)
        self.passengers = passengers
        self.size = size
        self.velocity = velocity      
        self.leave_stop_time = leave_stop_time #the planned time that the bus can leave its stop after dwelling 
        self.dispatch_time = dispatch_time #scheduled dispatching time        
        self.status = 0  #0 for inactive bus, 1 for moving bus, and 2 for dwelling bus, 3 for finished bus
        return
    def move(self):
        self.position += self.velocity * self.dt            
        return
    
class BusStop:
    def __init__(self, position, busstopID,arrival_rate,passengers):        
        self.busstopID = busstopID
        self.position = position        
        self.passengers = []
        self.arrival_rate = arrival_rate        
        self.headway = []  #store departure times of buses
        self.arrival_time = [0] #store arrival time of buses
        self.visited = []  #store all visited buses
    def passenger_arrival(self, dest,current_time):
        """Add a new passenger to the queue."""
        passenger = Passenger(origin=self.busstopID, dest=dest, start_time=current_time)
        self.passengers.append(passenger)    

class Model:
    def __init__(self,dt,NumberOfStop,LengthBetweenStop,EndTime,Headway,BurnIn,AlightTime,BoardTime,StoppingTime,MeanSpeed,Speed_std,Noise_std):        
        self.current_time = 0  # current time step of the simulation
        self.alighted_passengers = []
        self.busstops = []
        self.buses = []
        #Fixed parameters
        self.dt=dt
        self.GeoFence= dt * 14 + 5  #an area to tell if the bus reaches a bus stop
        self.NumberOfStop = NumberOfStop  
        self.LengthBetweenStop = LengthBetweenStop
        self.StopList = np.arange(0,NumberOfStop*LengthBetweenStop,LengthBetweenStop)      #this must be a list 
        self.EndTime = EndTime  #Last time in seconds
        self.Headway = Headway #scheduled time between each bus
        self.FleetSize = int(EndTime/Headway) #Total number of bus being dispatched from stop 0
        self.BurnIn = BurnIn # Burn-in time for passengers to arrive before the simulation begins        
        # Calibrating parameters
        self.AlightTime = AlightTime
        self.BoardTime = BoardTime
        self.StoppingTime = StoppingTime #some dead time to account for bus stopping and starting time after a bus
        self.MeanSpeed = MeanSpeed  #average speed of a bus at anytime
        self.Speed_std = Speed_std
        # Input parameters (also fixed, but we may get them from a data)
        self.ODTable = np.random.rand(self.NumberOfStop,self.NumberOfStop)  #a Origin-Destination table of passenger movements
        np.fill_diagonal(self.ODTable,0)  #the diagonal should be all zeros
        self.ArrivalRate = np.random.uniform(0.5/60,2/60,self.NumberOfStop) #randomise the arrival rate to each stop
        self.ArrivalRate[-1]=0 #no one should come to the last stop to board a bus!
        self.Noise_std = Noise_std #random Gaussian noise in meter
        # Initial Condition
        self.initialise_busstops()
        self.initialise_buses()
        return

    def step(self, random_order=False): #This is the main step function to move the model forward
        self.current_time += self.dt
        #print(self.current_time)        
        #STEP 1: loop through each bus stop and let passenger to arrive
        for busstop in self.busstops:            
            r=np.random.uniform()
            #probability of an arrival within the next second us F(x) = 1 - exp(-lambda)
            F = 1 - np.exp(-busstop.arrival_rate*self.dt)
            if r < F :
                #there is an arrival
                # generate the destination stop from the OD table
                if busstop.busstopID<self.NumberOfStop-2:
                    dest = int(np.random.choice(np.arange(busstop.busstopID+2,self.NumberOfStop),p=self.ODTable[busstop.busstopID+2:self.NumberOfStop,busstop.busstopID]/sum(self.ODTable[busstop.busstopID+2:self.NumberOfStop,busstop.busstopID])))                       
                else: dest = self.NumberOfStop
                #let a passenger arrives
                busstop.passenger_arrival(dest,self.current_time)
                
        #STEP 2: loop through each bus and let it moves or dwells
        for bus in self.buses:
           
           # CASE 1: INACTIVE BUS (not yet dispatched)
           if bus.status ==0: #inactive bus (not yet dispatched) 
              #check if it's time to dispatch yet?
              if self.current_time >= (bus.dispatch_time - self.dt): # if the bus is dispatched at the next time step
                 bus.status = 1 #change the status to moving bus
                 bus.velocity = np.random.normal(self.MeanSpeed/2,self.Speed_std/2)
                 #check if there is any passengers who are waiting for the bus at the first stop (Stop 0)                   
                 boarding_count = min(len(self.busstops[0].passengers),  (bus.size - len(bus.passengers)))
                 for p in self.busstops[0].passengers[1:boarding_count]:
                    p.time_waited_for_bus = self.current_time - p.start_time
                 bus.passengers.extend(self.busstops[0].passengers[1:boarding_count]) #add boarded passengers to the bus
                 self.busstops[0].passengers = self.busstops[0].passengers[boarding_count+1:]  #remove boarded passengers from the bus stop
                 #print('bus no: ',bus.busID, ' start moving at time: ',self.current_time)
           # CASE 2: MOVING BUS
           if bus.status == 1:  #moving bus
               bus.move()
               bus.velocity = np.random.normal(self.MeanSpeed,self.Speed_std)
               if bus.position > self.NumberOfStop*self.LengthBetweenStop: bus.status=3
               #if after moving, the bus enters a bus stop with passengers on it, then we move the status to dwelling
               dns = lambda x,y : abs(x-y)
               if  min(dns(self.StopList,bus.position)) <= self.GeoFence :  #reached a bus stop
                   Current_StopID=int(min(range(len(self.StopList)), key=lambda x:abs(self.StopList[x]-bus.position))) #find the nearest bus stop          
                   
                   if Current_StopID != bus.visited:  #if the bus hasn't visited this stop
                       bus.visited = Current_StopID
                       #check if there is any onboards passengers who wants to alight to that stop
                       out_passengers = [passenger for passenger in bus.passengers if passenger.dest==Current_StopID]                                                               
                       #check if there is any passengers who are waiting for the bus                   
                       boarding_count = min(len(self.busstops[Current_StopID].passengers),  (bus.size - len(bus.passengers)))
                       #keep a list of visited bus
                       self.busstops[Current_StopID].visited.extend([bus.busID]) #store the visited bus ID
    
                       #if there is no one boarding or alighting, the bus would just move on, so status is still 1. Otherwise the bus
                       #has to stop, which means status should be 2
                       if len(out_passengers)>0 or boarding_count>0: #if there is any one boarding or alighting
                           bus.status = 2  #change the status of the bus to dwelling
                           #print('Bus ',bus.busID, 'stopping at stop: ',Current_StopID,' at time:', self.current_time)                       
                           
                           if len(out_passengers)>0:
                               """Passengers that reached their destination leave the bus."""                                     
                               for passenger in out_passengers:
                                   #passenger.end_time = cur_time
                                   passenger.total_time = self.current_time - passenger.start_time                           
                                   self.alighted_passengers.append(passenger) #add the passengers to the list of alighted passengers
                                   bus.passengers.remove(passenger) #remove the passenger from the bus
                               #print('Alighting: ',len(out_passengers))
                           
                           if boarding_count>0:                        
                                """Passengers at the bus stop hop into the bus."""
                                for p in self.busstops[Current_StopID].passengers[1:boarding_count]:
                                    p.time_waited_for_bus = self.current_time - p.start_time
                                bus.passengers.extend(self.busstops[Current_StopID].passengers[1:boarding_count]) #add boarded passengers to the bus
                                self.busstops[Current_StopID].passengers = self.busstops[Current_StopID].passengers[boarding_count+1:]  #remove boarded passengers from the bus stop
                                #print('Boarding: ',boarding_count)
                                #the bus must only leave the stop when all the boarding and alighting is done
                                bus.leave_stop_time = self.current_time + len(out_passengers)*self.AlightTime + boarding_count*self.BoardTime + self.StoppingTime  #total time for dwelling
                       #store the headway and arrival times
                       self.busstops[Current_StopID].headway.extend([self.current_time-self.busstops[Current_StopID].arrival_time[-1]])  #store the headway to the previous bus
                       self.busstops[Current_StopID].arrival_time.extend([self.current_time])  #store the arrival time of the bus
            # CASE 3: DWELLING BUS (waiting for people to finish boarding and alighting)
           if bus.status == 2: 
              #check if people has finished boarding/alighting or not?
              if self.current_time >= (bus.leave_stop_time - self.dt): # if the bus hasn't left and can leave at the next time step
                 bus.status = 1 #change the status to moving bus
                 bus.velocity = np.random.normal(self.MeanSpeed/2,self.Speed_std/2)
           bus.states.append([bus.status,bus.position+np.random.normal(0,self.Noise_std),bus.velocity+ np.random.normal(0,self.Noise_std)]) #Now generate a noisy GPS signal of the bus
           bus.groundtruth.append([bus.status,bus.position,bus.velocity])
           #bus.GPS.extend(bus.position)
        return

    def initialise_busstops(self):
        for busstopID in range(len(self.StopList)):
            position = self.StopList[busstopID] #set up bus stop location
            arrival_rate = self.ArrivalRate[busstopID] #set up bus stop location
            passengers = []            
            #define the bus stop object and add to the model
            busstop = BusStop(position,busstopID,arrival_rate,passengers)
            #let some passengers arrive before the simulation even begins
            ArrivedPassengers=np.random.poisson(arrival_rate*self.BurnIn)
            for arrival in range(ArrivedPassengers):
                #there is an arrival
                if busstopID<self.NumberOfStop-2:
                    dest = int(np.random.choice(np.arange(busstopID+2,self.NumberOfStop),p=self.ODTable[busstopID+2:self.NumberOfStop,busstopID]/sum(self.ODTable[busstopID+2:self.NumberOfStop,busstopID])))                       
                else: dest = self.NumberOfStop
                #let a passenger arrives
                busstop.passenger_arrival(dest,0)
            self.add_busstops(busstop)
        return

    def initialise_buses(self):
        for busID in range(self.FleetSize):
            velocity = 0 # starting speed is 60 km/h ~ 16 m/s
            position = -self.MeanSpeed*self.dt #all buses starts at the first stop
            #next_stop=0
            passengers=[]
            size= 100 #vehicle size, fixed parameter
            total_passengers = 0
            leave_stop_time=9999 #this shouldn't matter but just in case
            status = 0  #bus started as inactive
            dispatch_time = busID*self.Headway  
            #define the bus object and add to the model
            bus = Bus(self.dt,busID,position,passengers,size,total_passengers,velocity,leave_stop_time,dispatch_time,status)            
            self.add_buses(bus)
            #print('Finished initialising the bus ',busID, ' at location: ', position, ' will dipatch at: ', dispatch_time)
        return
    
    def add_buses(self, bus):
        self.buses.append(bus)        
        return
    
    def add_busstops(self, busstop):
        self.busstops.append(busstop)        
        return

if True:  # Let's run the model    

    model_params = {
            "dt": 10,
            "NumberOfStop": 10,
            "LengthBetweenStop": 3000,
            "EndTime" : 7200,
            "Headway" : 10*60,
            "BurnIn" : 1 * 60,
            "AlightTime" : 1,
            "BoardTime": 3,
            "StoppingTime" : 3,
            "MeanSpeed" : 14,
            "Speed_std":3,
            "Noise_std" : 5            
            }
    '''
    Model runing and plotting
    '''
    model = Model(**model_params)
    _plot = True  # make _plot=True if you want to plot    

    if _plot == True:
        for time_step in range(int(model.EndTime/model.dt)):
            model.step()
            plt.text(0,0.015,'Number of waiting passengers')
            plt.plot(model.StopList,np.zeros_like(model.StopList),"|",markersize=20,alpha=0.3)
            plt.plot(model.StopList,np.zeros_like(model.StopList),alpha=0.3)        
            for busstop in model.busstops:
                val = 0. # this is the value where you want the data to appear on the y-axis.            
                plt.plot(busstop.position, np.zeros_like(busstop.position) + val, 'ok', alpha =0.2)
                plt.text(busstop.position, np.zeros_like(busstop.position) + val + 0.01,len(busstop.passengers))
            for bus in model.buses:            
                plt.plot(bus.position, np.zeros_like(bus.position) + val, '>',markersize=10)
                text0 = ['BusID= ',str(bus.busID+1),', occupancy= ',str(len(bus.passengers))]
                if bus.status!=0: plt.text(0, -bus.busID*0.003-0.01,"".join(text0))
            #plt.axis([-10, 31000, -0.1, 0.1])
            plt.axis('off')
            plt.pause(.001)
            plt.clf()
    else: 
        for time_step in range(int(model.EndTime/model.dt)): model.step()

if False: #plot buses GPS (after running)        
    GPSData = model.buses[0].GPS
    for b in range(1,model.FleetSize):
        GPSData=np.vstack((GPSData,model.buses[b].states))
        
    GPSData[GPSData<0]=0
    for l in range(model.FleetSize-1):
        plt.plot(GPSData[l])
        plt.xlabel('Time')
        plt.ylabel('Distance')
        
if False: #compile data to use laler
     
    StateData = model.buses[0].states
    for b in range(1,model.FleetSize):
        StateData=np.hstack((StateData,model.buses[b].states))
    StateData[StateData<0]=0

    GroundTruth = model.buses[0].groundtruth
    for b in range(1,model.FleetSize):
        GroundTruth=np.hstack((GroundTruth,model.buses[b].groundtruth))
    GroundTruth[GroundTruth<0]=0
    
    ArrivalData = model.busstops
    
    ArrivalData = ()
    for b in range(0,len(model.busstops)):
        ArrivalData+=tuple([model.busstops[b].arrival_time])
     
    import pickle

    # obj0, obj1, obj2 are created here...
    
    # Saving the objects:
    with open('BusSim_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([StateData, GroundTruth, ArrivalData,model.ODTable,model.ArrivalRate,model.ArrivalRate], f)
    
    # Copy this code to get back the objects:
    #with open('BusSim_data.pkl','rb') as f:  # Python 3: open(..., 'rb')
    #    StateData, GroundTruth, ArrivalData,ODTable,ArrivalRate,ArrivalRate = pickle.load(f)
    
    
   