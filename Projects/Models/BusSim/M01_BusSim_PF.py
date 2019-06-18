# -*- coding: utf-8 -*-
"""
This code will apply Particle Filter on BusSim, requires model BusSim_static.py
@author: geomlk
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
#from BusSim_static_deterministic import Bus,BusStop,Model
from BusSim_static_v2 import Bus,BusStop,Model

from ParticleFilter_MK import ParticleFilter
from copy import deepcopy
import pandas as pd


#Step 1: Load calibration results

def load_calibrated_params(IncreaseRate):
    
    #name0 = ['/Users/geomik/Dropbox/Minh_UoL/DA/ABM/BusSim/Calibration/BusSim_Model2_calibration_IncreaseRate_',str(IncreaseRate),'.pkl']
    name0 = ['C:/Users/geomlk/Dropbox/Minh_UoL/DA/ABM/BusSim/Calibration/BusSim_Model2_calibration_IncreaseRate_',str(IncreaseRate),'.pkl']
    str1 = ''.join(name0)    

    with open(str1, 'rb') as f:
    #with open('/Users/geomik/Dropbox/Minh_UoL/DA/ABM/BusSim/BusSim_model_calibration.pkl', 'rb') as f:      
        model_params, best_mean_model2,Sol_archived_mean,Sol_archived_std,PI_archived = pickle.load(f)
        
    #name0 = ['/Users/geomik/Dropbox/Minh_UoL/DA/ABM/BusSim/Calibration/BusSim_Model1_calibration_IncreaseRate_',str(IncreaseRate),'.pkl']
    name0 = ['C:/Users/geomlk/Dropbox/Minh_UoL/DA/ABM/BusSim/Calibration/BusSim_Model1_calibration_IncreaseRate_',str(IncreaseRate),'.pkl']
    str2 = ''.join(name0)    
        
    with open(str2, 'rb') as f:
    #with open('/Users/geomik/Dropbox/Minh_UoL/DA/ABM/BusSim/BusSim_model_calibration.pkl', 'rb') as f:      
        model_params, best_mean_model1,Sol_archived_mean,Sol_archived_std,PI_archived = pickle.load(f)

    return best_mean_model1,best_mean_model2    

def load_actual_params(IncreaseRate):
    #load up a model from a Pickle    
    #name0 = ['/Users/geomik/Dropbox/Minh_UoL/DA/ABM/BusSim/Data/Realtime_data_IncreaseRate_',str(IncreaseRate),'.pkl']
    name0 = ['C:/Users/geomlk/Dropbox/Minh_UoL/DA/ABM/BusSim/Data/Realtime_data_IncreaseRate_',str(IncreaseRate),'.pkl']
    str3 = ''.join(name0)    
    with open(str3,'rb') as f2:
        model_params,t0,x0,GroundTruth0 = pickle.load(f2)
    return model_params, t0,x0, GroundTruth0

def run_model(model_params):
    model = Model(model_params)
    
    RepGPS = []
    
    for time_step in range(int(model.EndTime / model.dt)):
        model.step()                
    
    RepGPS = model.buses[0].trajectory
    for b in range(1, model.FleetSize):
        RepGPS = np.vstack((RepGPS, model.buses[b].trajectory))
    RepGPS[RepGPS < 0] = 0        
    RepGPS=np.transpose(RepGPS)
    
    GroundTruth = pd.DataFrame(model.buses[0].groundtruth)
    for b in range(1, model.FleetSize):
        GroundTruth = np.hstack((GroundTruth, model.buses[b].groundtruth))
    GroundTruth[GroundTruth < 0] = 0

    return RepGPS, GroundTruth


if __name__ == '__main__':
    Results = []
    for IncreaseRate in (7,):
        model_params, t0,x0, GroundTruth0 = load_actual_params(IncreaseRate)
        best_mean_model1,best_mean_model2 = load_calibrated_params(IncreaseRate)
        
        '''
        Step 1:Run the actual parameters first to get one ground-truth trajectory
        '''    
        
        '''
        Step 1:Run the calibrated parameters and Particle Filter on it
        '''    
        ArrivalRate = best_mean_model2[0:model_params['NumberOfStop']]
        DepartureRate = best_mean_model2[model_params['NumberOfStop']:2*model_params['NumberOfStop']]
        TrafficSpeed = best_mean_model2[-2]
        maxDemand=best_mean_model2[-1]
        '''
        Step 2: Model runing and plotting
        '''
    
        model0=Model(model_params, TrafficSpeed,ArrivalRate,DepartureRate)
        t = np.arange(0, model_params['EndTime'], model_params['dt'])
        
        do_plot=True
        print('IncreaseRate = ',IncreaseRate)
        if do_plot==True:
            for p in (100,):
                for w in (1,):
        
                    filter_params = {
                        'number_of_particles': p,
                        'arr_std': 0,
                        'dep_std':0,
                        'traffic_std':0,
                        'resample_window': w,
                        'do_copies': True,
                        'do_save': True
                        }
                    model = deepcopy(model0)
                    pf = ParticleFilter(model, **filter_params)
                    plt.figure(3, figsize=(16 / 2, 9 / 2))
                    plt.clf()
                    plt.pause(5)
                    for niter in range(len(GroundTruth0)):
                        model.step()
                        true_state = GroundTruth0[niter]
                        measured_state = true_state #+ np.random.normal(0, 0., true_state.shape)  #to add noise in the measured_state if needed
                        pf.step(measured_state, true_state)
                        do_ani = True
                        if niter == len(GroundTruth0)-1 or do_ani:
                            plt.clf()
                            x = np.array([bus.trajectory for bus in pf.models[np.argmax(pf.weights)].buses]).T
                            t = np.arange(0, len(x)*model_params['dt'], model_params['dt'])
                            x[x <= 0] = np.nan
                            x[x >= model_params['NumberOfStop']*model_params['LengthBetweenStop']-1] = np.nan
                            
                            t1= t0[:len(t)-10]
                            x1 = x0[:len(t)-10,]                            
                            plt.plot(t1, x1,linewidth=1, color='k')
                            plt.plot(t, x, linewidth=2, linestyle='--')
                            plt.xlabel('Time (s)')
                            plt.ylabel('Distance (m)')
                            axes = plt.gca()
                            axes.set_ylim([0,model_params['NumberOfStop']*model_params['LengthBetweenStop']+10]) 
                            axes.set_xlim([0,model_params['EndTime']]) 
                            plt.plot([], [], linewidth=2,linestyle = '--',label='BusSim prediction')
                            plt.plot([], [], linewidth=1,color='black',linestyle = '-',label='Synthetic Real-time data')        
                            plt.legend(loc='upper left')
                            if do_ani:
                                plt.pause(0.1/30)
                                plt.show()
                    plt.show()
                    plt.savefig('Fig_PF_std0.0005_bestparticle.pdf', dpi=200,bbox_inches='tight')
    
        else:
            
            filter_params = {
                'number_of_particles': 500,
                'arr_std': 0.001,
                'dep_std':0.001,
                'traffic_std':0.001,
                'resample_window': 1,
                'do_copies': True,
                'do_save': True
                }
            model = deepcopy(model0)
            pf = ParticleFilter(model, **filter_params)
    
            for niter in range(len(GroundTruth0)):
                model.step()
                true_state = GroundTruth0[niter]
                measured_state = true_state #+ np.random.normal(0, 0., true_state.shape)  #to add noise in the measured_state if needed
                pf.step(measured_state, true_state)
            x = np.array([bus.trajectory for bus in pf.models[np.argmax(pf.weights)].buses]).T
            x[x <= 0] = 0
            x[x >= model_params['NumberOfStop']*model_params['LengthBetweenStop']-1] = 0
            x0[np.isnan(x0)]=0
            Results.append(np.mean(np.abs(x-x0)))
            
                
        plot_calibrated_vs_grouthtruth=False
        if plot_calibrated_vs_grouthtruth:
            #run calibrated model once and see if it's deviated from groundtruth
            CalGPS,CalGroundTruth  = run_model(model_params)                    
            x = GroundTruth0[:, 1::4]
            x[x <= 0] = np.nan
            x[x >= model_params['NumberOfStop']*model_params['LengthBetweenStop']-1] = np.nan
            plt.plot(t, x, 'k')
            x2 = CalGroundTruth[:, 1::4]
            x2[x2 <= 0] = np.nan
            x2[x2 >= model_params['NumberOfStop']*model_params['LengthBetweenStop']-1] = np.nan
            plt.plot(t, x2,linestyle='--')
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (m)')
     
        


















