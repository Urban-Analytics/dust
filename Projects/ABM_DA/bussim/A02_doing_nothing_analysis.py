# -*- coding: utf-8 -*-
"""
This code will analyse the modelling results of BusSim and plot it along side a number of uncalibrated models (timespace diagram of bus trajectories)

@author: geomlk
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
os.chdir("/Users/minhkieu/Documents/Github/dust/Projects/ABM_DA/bussim/")

#Step 1: Load calibration results
def load_actual_params_IncreaseRate(IncreaseRate):
    #load up a model from a Pickle    
    #with open('C:/Users/geomlk/Dropbox/Minh_UoL/DA/ABM/BusSim/Data/Realtime_data_IncreaseRate_9.pkl','rb') as f2:
    name0 = ['./Data/Realtime_data_IncreaseRate_',str(IncreaseRate),'.pkl']
    str1 = ''.join(name0)    

    with open(str1, 'rb') as f:
        model_params,t,x,GroundTruth = pickle.load(f)
    return model_params,t,x

def load_actual_params_maxDemand(maxDemand):
    #load up a model from a Pickle    
    #with open('C:/Users/geomlk/Dropbox/Minh_UoL/DA/ABM/BusSim/Data/Realtime_data_IncreaseRate_9.pkl','rb') as f2:
    name0 = ['./Data/Realtime_data_static_maxDemand_',str(maxDemand),'.pkl']
    str1 = ''.join(name0)    

    with open(str1, 'rb') as f:
        model_params,t,x,GroundTruth = pickle.load(f)
    return model_params,t,x

def rmse(yhat,y):    #define the RMSE function
    return np.sqrt(np.square(np.subtract(yhat, y).mean()))

def IncreaseRate_analysis(): #this is the code to analyse the simulation results when the demand increase by 1 to 20%
    Results = [0,0]
    do_plot=True
    for IncreaseRate in range(1,20,1):
        #load the synthetic real-time GPS
        model_params, t,x = load_actual_params_IncreaseRate(IncreaseRate)
        #define parameters for simulation
        NumberOfStop=20
        minDemand=0.5
        maxDemand=2
        #Initialise the ArrivalRate and DepartureRate
        ArrivalRate = np.random.uniform(minDemand / 60, maxDemand / 60, NumberOfStop)
        DepartureRate = np.sort(np.random.uniform(0.05, 0.5,NumberOfStop))
        DepartureRate[0]=0
        TrafficSpeed = np.random.uniform(11, 17)   
        #Initialise the model parameters
        model_params = {
            "dt": 10,
            "minDemand":minDemand,        
            "NumberOfStop": NumberOfStop,
            "LengthBetweenStop": 2000,
            "EndTime": 6000,
            "Headway": 5 * 60,
            "BurnIn": 1 * 60,
            "AlightTime": 1,
            "BoardTime": 3,
            "StoppingTime": 3,
            "BusAcceleration": 3  # m/s          
        }
        '''run BusSim-deterministic with random parameters'''
        from BusSim_deterministic import Model as Model1
        model = Model1(model_params, TrafficSpeed,ArrivalRate,DepartureRate)    
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x3 = np.array([bus.trajectory for bus in model.buses]).T        
        t3 = np.arange(0, model.EndTime, model.dt)
        x3[x3 <= 0 ] = np.nan
        x3[x3 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            
        '''run BusSim-stochastic with random parameters'''
        ArrivalRate = np.random.uniform(minDemand / 60, maxDemand / 60, NumberOfStop)
        DepartureRate = np.sort(np.random.uniform(0.05, 0.5,NumberOfStop))
        DepartureRate[0]=0
        TrafficSpeed = np.random.uniform(11, 17)           
        from BusSim_stochastic import Model as Model2
        model = Model2(model_params, TrafficSpeed,ArrivalRate,DepartureRate)    
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x2 = np.array([bus.trajectory for bus in model.buses]).T        
        t2 = np.arange(0, model.EndTime, model.dt)
        x2[x2 <= 0 ] = np.nan
        x2[x2 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            
        ''' we may plot individual run if it's needed'''
        if do_plot:
            plt.figure(3, figsize=(16 / 2, 9 / 2))
            plt.clf()            
            plt.ylabel('Distance (m)')
            plt.xlabel('Time (s)')
            plt.plot(t3, x3, linewidth=1.5,linestyle = '--',color='b')
            plt.plot(t2, x2, linewidth=1.5,linestyle = ':',color='r')
            plt.plot(t, x, linewidth=1,color='black',linestyle = '-')
            plt.plot([], [], linewidth=1.5,linestyle = '--',color='b',label='BusSim-deterministic')
            plt.plot([], [], linewidth=1.5,linestyle = ':',color='r',label='BusSim-stochastic')
            plt.plot([], [], linewidth=1,color='black',linestyle = '-',label='Real-time')
            plt.legend()
            plt.show()
            name0 = ['./Figures/Fig_do_nothing_IncreaseRate_',str(IncreaseRate),'.pdf']
            str1 = ''.join(name0)    
            plt.savefig(str1, dpi=200,bbox_inches='tight')
        '''collect outputs data and calculate RMSE'''
        x3[np.isnan(x3)]=0
        x2[np.isnan(x2)]=0
        x[np.isnan(x)]=0
        RMSE1 = rmse(x3,x)
        RMSE2 = rmse(x2,x)
        Results =  np.vstack((Results,[RMSE1,RMSE2]))
    ''' this plot is the main results plot'''        
    do_plot_results=True
    if do_plot_results:
        plt.figure(3, figsize=(16 / 2, 9 / 2))
        plt.clf()            
        plt.plot(np.arange(1,20,1),Results[1:,0],linewidth=1.5,linestyle = '--',color='b',label='BusSim-deterministic')
        plt.plot(np.arange(1,20,1),Results[1:,1],linewidth=1.5,linestyle = ':',color='r',label='BusSim-stochastic')
        plt.ylabel('RMSE (m)')
        plt.xlabel(r'$\xi$ (%)')
 
        plt.legend()
        plt.show()
        plt.savefig('./Figures/Fig_do_nothing_results.pdf', dpi=200,bbox_inches='tight')

    return Results

'''
Function to evaluate results when the maximum demand increases from 0.5 to 4.5
'''
def maxDemand_analysis():
    Results = [0,0]
    do_plot=False
    for maxDemand in range(1,10,1):
        maxDemand =maxDemand/2
    
        model_params, t,x = load_actual_params_maxDemand(maxDemand)
    
        NumberOfStop=20
        minDemand=0.5
        
        #Initialise the ArrivalRate and DepartureRate
        ArrivalRate = np.random.uniform(minDemand / 60, maxDemand / 60, NumberOfStop)
        DepartureRate = np.sort(np.random.uniform(0.05, 0.5,NumberOfStop))
        DepartureRate[0]=0
        TrafficSpeed = np.random.uniform(11, 17)   
        #Initialise the model parameters
        model_params = {
            "dt": 10,
            "minDemand":minDemand,        
            "NumberOfStop": NumberOfStop,
            "LengthBetweenStop": 2000,
            "EndTime": 6000,
            "Headway": 5 * 60,
            "BurnIn": 1 * 60,
            "AlightTime": 1,
            "BoardTime": 3,
            "StoppingTime": 3,
            "BusAcceleration": 3  # m/s          
        }
        '''run BusSim-deterministic with random parameters'''    
        from BusSim_deterministic import Model as Model1
        model = Model1(model_params, TrafficSpeed,ArrivalRate,DepartureRate)    
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x3 = np.array([bus.trajectory for bus in model.buses]).T        
        t3 = np.arange(0, model.EndTime, model.dt)
        x3[x3 <= 0 ] = np.nan
        x3[x3 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            

        ArrivalRate = np.random.uniform(minDemand / 60, maxDemand / 60, NumberOfStop)
        DepartureRate = np.sort(np.random.uniform(0.05, 0.5,NumberOfStop))
        DepartureRate[0]=0
        TrafficSpeed = np.random.uniform(11, 17)   
        '''run BusSim-stochastic with random parameters'''
    
        from BusSim_stochastic import Model as Model2
        model = Model2(model_params, TrafficSpeed,ArrivalRate,DepartureRate)    
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x2 = np.array([bus.trajectory for bus in model.buses]).T        
        t2 = np.arange(0, model.EndTime, model.dt)
        x2[x2 <= 0 ] = np.nan
        x2[x2 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            
        ''' we may plot individual run if it's needed'''

        if do_plot:
            plt.figure(3, figsize=(16 / 2, 9 / 2))
            plt.clf()            
            plt.ylabel('Distance (m)')
            plt.xlabel('Time (s)')
            plt.plot(t3, x3, linewidth=.5,linestyle = '--',color='b')
    
            plt.plot(t2, x2, linewidth=1,linestyle = ':',color='r')
            plt.plot(t, x, linewidth=1,color='black',linestyle = '-')
            
            plt.plot([], [], linewidth=.5,linestyle = '--',color='b',label='BusSim-deterministic')
            plt.plot([], [], linewidth=1,linestyle = ':',color='r',label='BusSim-stochastic')
            plt.plot([], [], linewidth=1,color='black',linestyle = '-',label='Real-time')
        
            plt.legend()
            plt.show()
            name0 = ['./Figures/Fig_do_nothing_maxDemand_',str(maxDemand),'.pdf']
            str1 = ''.join(name0)    
    
            plt.savefig(str1, dpi=200,bbox_inches='tight')
        '''collect outputs data and calculate RMSE'''
        
        x3[np.isnan(x3)]=0
        x2[np.isnan(x2)]=0
        x[np.isnan(x)]=0
        RMSE1 = rmse(x3,x)
        RMSE2 = rmse(x2,x)
        Results =  np.vstack((Results,[RMSE1,RMSE2]))
    ''' this plot is the main results plot'''        
        
    do_plot_results=True
    if do_plot_results:
        plt.figure(3, figsize=(16 / 2, 9 / 2))
        plt.clf()            
        plt.plot(np.arange(1,10,1),Results[1:,0],linewidth=1.5,linestyle = '--',color='b',label='BusSim-deterministic')
        plt.plot(np.arange(1,10,1),Results[1:,1],linewidth=1.5,linestyle = ':',color='r',label='BusSim-stochastic')
        plt.ylabel('RMSE (m)')
        plt.xlabel(r'$maxDemand$ (passenger/min)')
        plt.xticks(np.arange(1,10,1), (np.arange(1,10,1)/2))
 
        plt.legend()
        plt.show()
        plt.savefig('./Figures/Fig_do_nothing_results_maxDemand.pdf', dpi=200,bbox_inches='tight')

    return Results    
if __name__ == '__main__': #main function, just call one of the two evaluation
    
    #Results=maxDemand_analysis()
    Results=IncreaseRate_analysis()
    
    
    