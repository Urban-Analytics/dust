"""
This function calibrates the BusSim models against the data generated from BusSim-truth

The calibration technique is Cross-Entropy Method (https://en.wikipedia.org/wiki/Cross-entropy_method)

Written by: Minh Kieu, University of Leeds
"""
import numpy as np
import pickle
from scipy.stats import truncnorm
import os
os.chdir("/Users/minhkieu/Documents/Github/dust/Projects/ABM_DA/bussim/")

# COMMENT/UNCOMMENT the model you want to calibrate here: 

#from BusSim_deterministic import Model
from BusSim_stochastic import Model

def unpickle_initiation(maxDemand):
    #load up ground truth data from a pickle
    # Remember to run BusSim_static_v2 first to generate Historical_data_static
    name0 = ['./Data/Historical_data_static_maxDemand_',str(maxDemand),'.pkl']
    str1 = ''.join(name0)    
    with open(str1, 'rb') as f:
        model_params,meanGPS,stdGPS = pickle.load(f)
    return model_params, meanGPS,stdGPS

def solution2model(model_params,solution):
    #apply a solution to a model
    ArrivalRate = solution[0:(model_params['NumberOfStop'])]
    DepartureRate = solution[model_params['NumberOfStop']:(2*model_params['NumberOfStop'])]    
    TrafficSpeed = solution[-1]
    #load model
    model = Model(model_params, TrafficSpeed,ArrivalRate,DepartureRate)
    return model

def run_model(model):
    '''
    Return the GPS data for that particular run of the model
    '''    
    RunGPS = []
    
    for time_step in range(int(model.EndTime / model.dt)):
        model.step()
    RunGPS = model.buses[0].trajectory
    for b in range(1, model.FleetSize):
        RunGPS = np.vstack((RunGPS, model.buses[b].trajectory))
    RunGPS[RunGPS < 0] = 0        
    RunGPS=np.transpose(RunGPS)
    return RunGPS

if __name__ == '__main__':  # Let's run the model
    '''
    Step 1: Model initiation and starting solution
    '''   
    CEM_params = { 
            'MaxIteration':50, #maximum number of iteration
            'NumSol': 500,  #number of solutions being evaluated at each iteration
            'NumRep': 15, #number of replication each time we run the model            
            'Elite': 0.10  #pick 15% of best solution for the next iteration
            }    
    TrafficSpeed_std=1.5
    TrafficSpeed_init=14
    
    for maxDemand in range(1,10,2):
        maxDemand = maxDemand/2
        #load model parameters
        print('maxDemand  = ', maxDemand)
        model_params,meanGPS,stdGPS = unpickle_initiation(maxDemand)        
        
        #Starting solution   
        mean_arr = np.random.uniform(model_params['minDemand'] / 60, maxDemand/60, model_params['NumberOfStop']-2)
        std_arr = 0.025*np.ones(model_params['NumberOfStop']-2)
        mean_dep = np.sort(np.random.uniform(0.05, 0.5,model_params['NumberOfStop']-3))
        std_dep = 0.05*np.ones(model_params['NumberOfStop']-3)
        mean_traffic = np.random.uniform(TrafficSpeed_init - TrafficSpeed_std, TrafficSpeed_init + TrafficSpeed_std)
        std_traffic = TrafficSpeed_std
        '''
        Step 2: Main CEM loop for model calibration
        '''
        Sol_archived_mean = []
        Sol_archived_std = []
        best_PI=0
        PI_archived =[]
        last5best = np.zeros(5)
        for ite in range(CEM_params['MaxIteration']):
            '''
            Step 2.1: Generate solution
            '''
            print('Iteration number ',ite)        
            Sol_arr = np.zeros([CEM_params['NumSol'],1])
            for p in range(len(mean_arr)):            
                lower = 0             
                mu = mean_arr[p]
                sigma = std_arr[p]            
                upper = mu + 2*sigma
                temp_arr=truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=CEM_params['NumSol'])
                Sol_arr=np.append(Sol_arr,temp_arr[:,None],axis=1)
            Sol_arr = np.hstack([Sol_arr,np.zeros([CEM_params['NumSol'],1])])
            
            Sol_dep = np.zeros([CEM_params['NumSol'],2])
            for p in range(model_params['NumberOfStop']-3):            
                lower = 0
                mu = mean_dep[p]
                sigma = std_dep[p]            
                upper = mu + 2*sigma
                temp_arr=truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=CEM_params['NumSol'])
                Sol_dep=np.append(Sol_dep,temp_arr[:,None],axis=1)
            Sol_dep = np.hstack([Sol_dep,np.ones([CEM_params['NumSol'],1])])
            
            mu = mean_traffic
            sigma = std_traffic 
            lower = mu - 2*sigma
            upper = mu + 2*sigma
            Sol_traffic=truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=CEM_params['NumSol'])
            
            #combine the solutions
            Sol0 = np.hstack([Sol_arr,Sol_dep,Sol_traffic[:,None]])    
            
            '''
            Step 2.2: Evaluate each solution in Sol0
            '''
    
            PIm = []
            for m in range(len(Sol0)):      
                #build up model
                SolGPS=[]
                for r in range(CEM_params['NumRep']):
                    model = solution2model(model_params,Sol0[m])        
                    RunGPS = run_model(model)             
                    SolGPS.append(RunGPS)            
                meanSolGPS = np.mean(SolGPS,axis=0)
                stdSolGPS = np.std(SolGPS,axis=0)
                PIm.extend([np.mean(np.abs(meanSolGPS-meanGPS))+np.mean(np.abs(stdSolGPS-stdGPS))])
            Elist = np.argsort(PIm)[0:(int(CEM_params['Elite']*CEM_params['NumSol']))]
            print('Current best solution: ', PIm[Elist[0]]) #best solution
                           
            '''
            Step 2.3: Generate new solutions for the next iteration
            '''
            mean_arr=np.mean(Sol_arr[Elist],axis=0)[1:model_params['NumberOfStop']-1]
            std_arr = np.std(Sol_arr[Elist],axis=0)[1:model_params['NumberOfStop']-1]
            mean_dep = np.mean(Sol_dep[Elist],axis=0)[2:model_params['NumberOfStop']-1]
            std_dep = np.std(Sol_dep[Elist],axis=0)[2:model_params['NumberOfStop']-1]
            mean_traffic = np.mean(Sol_traffic[Elist],axis=0)            
            std_traffic = np.std(Sol_traffic[Elist],axis=0)            
            
            '''
            Step 2.4: store the best solution and the current mean & std of the current generation of solutions
            '''
            if PIm[Elist[0]] > best_PI:
                best_PI = PIm[Elist[0]]
                best_mean = Sol0[Elist[0]]          
            sol_mean = np.concatenate([mean_arr,mean_dep,[mean_traffic]])
            sol_std = np.concatenate([std_arr,std_dep,[std_traffic]])
            Sol_archived_mean.append(sol_mean)
            Sol_archived_std.append(sol_std)
            PI_archived.append(PIm[Elist[0]])
            
            if abs((best_PI-np.mean(last5best))/best_PI) < 0.05 and abs((best_PI-last5best[-1])/best_PI) < 0.05: #break the loop if the best mean doesn't change more than 5% compared to the average best PI
                '''
                Step 3: Store the final parameter solution of BusSim                            '''
                
                name1 = ['./Calibration/BusSim_Model2_calibration_static_maxDemand_',str(maxDemand),'.pkl']
                str2 = ''.join(name1)
                with open(str2, 'wb') as f:
                    pickle.dump([model_params, best_mean,Sol_archived_mean,Sol_archived_std,PI_archived], f)
                print('Output to file: ',str2)    
                break
            else: 
                last5best = np.concatenate([last5best[1:],[0]]) #otherwise keep storing the best_PI            
                last5best[-1]=best_PI
            

    
   
    
    
