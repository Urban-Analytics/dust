# -*- coding: utf-8 -*-
"""
This code evaluates the outputs from calibrated BusSim
@author: geomlk
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
os.chdir("/Users/minhkieu/Documents/Github/dust/Projects/ABM_DA/bussim/")


'''
Step 1: Load calibration results
'''
def load_calibrated_params_IncreaseRate(IncreaseRate):
    name0 = ['./Calibration/BusSim_Model2_calibration_IncreaseRate_',str(IncreaseRate),'.pkl']
    str1 = ''.join(name0)    
    with open(str1, 'rb') as f:      
        model_params, best_mean_model2,Sol_archived_mean,Sol_archived_std,PI_archived = pickle.load(f)

    name0 = ['./Calibration/BusSim_Model1_calibration_IncreaseRate_',str(IncreaseRate),'.pkl']
    str1 = ''.join(name0)    
    with open(str1, 'rb') as f:      
        model_params, best_mean_model1,Sol_archived_mean,Sol_archived_std,PI_archived = pickle.load(f)
    return best_mean_model1,best_mean_model2    

def load_calibrated_params_maxDemand(maxDemand):
    name0 = ['./Calibration/BusSim_Model2_calibration_static_maxDemand_',str(maxDemand),'.pkl']
    str1 = ''.join(name0)    
    with open(str1, 'rb') as f:      
        model_params, best_mean_model2,Sol_archived_mean,Sol_archived_std,PI_archived = pickle.load(f)

    name0 = ['./Calibration/BusSim_Model1_calibration_static_maxDemand_',str(maxDemand),'.pkl']
    str1 = ''.join(name0)    
    with open(str1, 'rb') as f:      
        model_params, best_mean_model1,Sol_archived_mean,Sol_archived_std,PI_archived = pickle.load(f)
    return best_mean_model1,best_mean_model2    
'''
Step 2: Load synthetic real-time data
'''
def load_actual_params_IncreaseRate(IncreaseRate):
    #load up a model from a Pickle    
    name0 = ['./Data/Realtime_data_IncreaseRate_',str(IncreaseRate),'.pkl']
    str1 = ''.join(name0)    

    with open(str1, 'rb') as f:
        model_params,t,x,GroundTruth = pickle.load(f)
    return model_params,t,x

def load_actual_params_maxDemand(maxDemand):
    #load up a model from a Pickle    
    name0 = ['./Data/Realtime_data_static_maxDemand_',str(maxDemand),'.pkl']
    str1 = ''.join(name0)    

    with open(str1, 'rb') as f:
        model_params,t,x,GroundTruth = pickle.load(f)
    return model_params,t,x
#define RMSE function
def rmse(yhat,y):
    return np.sqrt(np.square(np.subtract(yhat, y).mean()))
'''
Step 3: Evaluation of calibrated models when the arrival rate is changing by 1 to 20%
'''
def IncreaseRate_analysis():
    Results = [0,0]
    do_plot=True
    for IncreaseRate in range(1,20,2):
        #load real-time data
        model_params, t,x = load_actual_params_IncreaseRate(IncreaseRate)
        #load calibrated parameters
        best_mean_model1,best_mean_model2 = load_calibrated_params_IncreaseRate(IncreaseRate)
        #load the BusSim-static model
        from BusSim_stochastic import Model as Model2
        ArrivalRate = best_mean_model2[0:(model_params['NumberOfStop'])]
        DepartureRate = best_mean_model2[model_params['NumberOfStop']:(2*model_params['NumberOfStop'])]    
        TrafficSpeed = best_mean_model2[-2]
        #load model
        model = Model2(model_params, TrafficSpeed,ArrivalRate,DepartureRate)
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x2 = np.array([bus.trajectory for bus in model.buses]).T        
        t2 = np.arange(0, model.EndTime, model.dt)
        x2[x2 <= 0 ] = np.nan
        x2[x2 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            
        
        #load the BusSim-stochastic model
        from BusSim_deterministic import Model as Model1
        ArrivalRate = best_mean_model1[0:(model_params['NumberOfStop'])]
        DepartureRate = best_mean_model1[model_params['NumberOfStop']:(2*model_params['NumberOfStop'])]    
        TrafficSpeed = best_mean_model1[-2]
        #load model
        model = Model1(model_params, TrafficSpeed,ArrivalRate,DepartureRate)
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x3 = np.array([bus.trajectory for bus in model.buses]).T        
        t3 = np.arange(0, model.EndTime, model.dt)
        x3[x3 <= 0 ] = np.nan
        x3[x3 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            
        #plot individual run (if needed)
        if do_plot:
            plt.figure(3, figsize=(16 / 2, 9 / 2))
            plt.clf()            
            plt.plot(t, x, linewidth=1,color='black',linestyle = '-')
            plt.plot(t2, x2, linewidth=1.5,linestyle = ':',color='r')
            plt.ylabel('Distance (m)')
            plt.xlabel('Time (s)')
            plt.plot(t3, x3, linewidth=1.5,linestyle = '--',color='b')

            plt.plot([], [], linewidth=1.5,linestyle = ':',color='r',label='BusSim-stochastic')
            plt.plot([], [], linewidth=1.5,linestyle = '--',color='b',label='BusSim-deterministic')
            plt.plot([], [], linewidth=1,color='black',linestyle = '-',label='Real-time')
    
            plt.legend()
            plt.show()
            name0 = ['./Figures/Fig_calibration_IncreaseRate_',str(IncreaseRate),'.pdf']
            str1 = ''.join(name0)    
            plt.savefig(str1, dpi=200,bbox_inches='tight')
        #calculate RMSE    
        x3[np.isnan(x3)]=0
        x2[np.isnan(x2)]=0
        x[np.isnan(x)]=0
        RMSE1 = rmse(x3,x)
        RMSE2 = rmse(x2,x)
        Results =  np.vstack((Results,[RMSE1,RMSE2]))
    #plot the evaluation results        
    do_plot_results=True
    if do_plot_results:
        plt.figure(3, figsize=(16 / 2, 9 / 2))
        plt.clf()            
        plt.plot(np.arange(1,20,2),Results[1:,1],linewidth=1.5,linestyle = '--',color='b',label='BusSim-deterministic')
        plt.plot(np.arange(1,20,2),Results[1:,0],linewidth=1.5,linestyle = ':',color='r',label='BusSim-stochastic')
        plt.ylabel('RMSE (m)')
        plt.xlabel(r'$\xi$ (%)') 
        plt.legend()
        plt.show()
        plt.savefig('./Figures/Fig_calibration_results_IncreaseRate.pdf', dpi=200,bbox_inches='tight')

    return Results
'''
Step 3: Evaluation of the case when the maxDemand increases from 0.5 to 4.5
'''
def maxDemand_analysis():
    Results = [0,0]
    do_plot=False
    for maxDemand in range(1,10,2):
        maxDemand=maxDemand/2
        model_params, t,x = load_actual_params_maxDemand(maxDemand)
        best_mean_model1,best_mean_model2 = load_calibrated_params_maxDemand(maxDemand)
        from BusSim_stochastic import Model as Model2
        ArrivalRate = best_mean_model2[0:(model_params['NumberOfStop'])]
        DepartureRate = best_mean_model2[model_params['NumberOfStop']:(2*model_params['NumberOfStop'])]    
        TrafficSpeed = best_mean_model2[-1]
        #load model
        model = Model2(model_params, TrafficSpeed,ArrivalRate,DepartureRate)
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x2 = np.array([bus.trajectory for bus in model.buses]).T        
        t2 = np.arange(0, model.EndTime, model.dt)
        x2[x2 <= 0 ] = np.nan
        x2[x2 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            
        from BusSim_deterministic import Model as Model1
        ArrivalRate = best_mean_model1[0:(model_params['NumberOfStop'])]
        DepartureRate = best_mean_model1[model_params['NumberOfStop']:(2*model_params['NumberOfStop'])]    
        TrafficSpeed = best_mean_model1[-1]
        #load model
        model = Model1(model_params, TrafficSpeed,ArrivalRate,DepartureRate)
        for time_step in range(int(model.EndTime / model.dt)):
            model.step()
        x3 = np.array([bus.trajectory for bus in model.buses]).T        
        t3 = np.arange(0, model.EndTime, model.dt)
        x3[x3 <= 0 ] = np.nan
        x3[x3 >= (model.NumberOfStop * model.LengthBetweenStop)] = np.nan            
        #plot individual plots if it's needed
        if do_plot:
            plt.figure(3, figsize=(16 / 2, 9 / 2))
            plt.clf()            
            plt.plot(t2, x2, linewidth=1,linestyle = ':',color='r',label='BusSim-stochastic')
            plt.plot(t, x, linewidth=1,color='black',linestyle = '-',label='Real-time')
            plt.ylabel('Distance (m)')
            plt.xlabel('Time (s)')
            plt.plot(t3, x3, linewidth=.5,linestyle = '--',color='b',label='BusSim-deterministic')
            plt.legend()
            plt.show()
            name0 = ['./Figures/Fig_calibration_maxDemand_',str(maxDemand),'.pdf']
            str1 = ''.join(name0)    
            plt.savefig(str1, dpi=200,bbox_inches='tight')
        #calculate RMSE for each run    
        x3[np.isnan(x3)]=0
        x2[np.isnan(x2)]=0
        x[np.isnan(x)]=0
        RMSE1 = rmse(x3,x)
        RMSE2 = rmse(x2,x)
        Results =  np.vstack((Results,[RMSE1,RMSE2]))
    #plot the evaluation results        
    do_plot_results=True
    if do_plot_results:
        plt.figure(3, figsize=(16 / 2, 9 / 2))
        plt.clf()            
        plt.plot(np.arange(1,10,2),Results[1:,0],linewidth=1.5,linestyle = '--',color='b',label='BusSim-deterministic')
        plt.plot(np.arange(1,10,2),Results[1:,1],linewidth=1.5,linestyle = ':',color='r',label='BusSim-stochastic')
        plt.ylabel('RMSE (m)')
        plt.xlabel(r'$maxDemand$ (passenger/min)')
        plt.xticks(np.arange(1,10,2), (np.arange(1,10,2)/2)) 
        plt.legend()
        plt.show()
        plt.savefig('./Figures/Fig_calibration_results_maxDemand.pdf', dpi=200,bbox_inches='tight')

    return Results

if __name__ == '__main__':  #main code, just call the evaluation codes
    
    Results = IncreaseRate_analysis()
    #Results = maxDemand_analysis()