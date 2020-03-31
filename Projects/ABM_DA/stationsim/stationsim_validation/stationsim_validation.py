#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:49:24 2020

@author: RC

Validation test for comparing model outputs from python and cpp stationsim.
We do this as follows:
- Generate two random samples of models 
- Calculate corresponding Ripley's K (RK) curves for each model
- Generate a data frame for the two populations of RK curves
- Save said frame and analyse using panel regression in R.
- Analysis determines whether the two groups are statistically indistinguishable.
"""

import numpy as np
import sys
import os
import multiprocessing
from astropy.stats import RipleysKEstimator # astropy's ripley's K
import pandas as pd
import glob
import datetime

sys.path.append("..")
from stationsim_model import Model #python version of stationsim

import matplotlib.pyplot as plt
from seaborn import kdeplot

class HiddenPrints:
    
    """stop unnecessary printing from stationsim 
    
    We get a lot of `iterations : X` prints for a large number of 
    stationsim runs. This stops the printing for tidiness.
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
        
class stationsim_RipleysK():
    
    """ Class for calculating Ripley' K curves on stationsim collisions
    and saving them as pandas dataframes.
    """

    def generate_Model_Sample(self, n_runs, model_params, single_process = False):
    
        
        """ function for generating stationsim model runs to test
        
        Parameters
        ------
        n_runs : int
            `n_runs`number of stationsim runs to generate. 
            Must be positive intiger.
            
        model_params : dict
            `model_params` dictionary of model parameters  required for 
            stationsim to run. See stationsim_model.py for more details.
            
        single_process : bool (default False)
             whether to run the models as a single process or using
             multiple processes simultaneously.
            
        Returns
        ------
        models : list
            a list of completed stationsim `models` given the required
            width, height, pop_total, and gate_speed.
        """
        
        #placeholder list
        models = []
        
        if n_runs > 1 and model_params["random_seed"] != None:
            raise Exception("Error: the 'random_seed' parameter is not None\
            which means that all models generate the same results, which\
            I'm sure isn't what you want!")
        elif n_runs < 1:
            raise Exception("Error: need one or more 'n_runs', not {}".format(n_runs))
        
        #supress excessive printing
        with HiddenPrints():
            if single_process or n_runs == 1:
                for _ in range(n_runs):
                    #generate model and run til status goes back to 0 (finished)
                    model = Model(**model_params)
                    while model.status == 1:
                        model.step()
                    models.append(model)
            else:
                pool = multiprocessing.Pool()
                try:
                    numcores = multiprocessing.cpu_count()
                    models = pool.map(stationsim_RipleysK.run_model, [model_params for _ in range(n_runs)])
                finally: 
                    pool.close() # Make sure whatever happens the processes are killed
            
        return models
    
    @staticmethod
    def run_model(model_params):
        
        """
        Create a new stationsim model using `model_params` and step it
        until it has finished.
        
        
        Parameters
        ------
        model_params : dict
            `model_params` dictionary of model parameters  required for 
            stationsim to run. See stationsim_model.py for more details.
            
        Returns
        ------
        model : StaionSim object
            the finished model
        """
        model = Model(**model_params)
        while model.status == 1:
            model.step()
        return model
        

    def ripleysKE(self, collisions, width, height):
        
        """ Generate Ripley's K (RK) curve for collisions in StationSim region.
        
        For more info on RK see:
        https://docs.astropy.org/en/stable/stats/ripley.html
        https://wiki.landscapetoolbox.org/doku.php/spatial_analysis_methods:ripley_s_k_and_pair_correlation_function"
        
        
        Parameters
        ------
        
        collisions : list
            list of model `collisions`
            
        width, height : float
            `width` and `height` of stationsim model
        
        Returns 
        ------
        
        rkes, rs : list
            lists of radii `rs` and corresponding Ripley's K values `rkes`
            for a given set of model collisions.
            
        """
        
        "define area of stationsim."
        
        "init astropy RKE class with stationsim boundaries/area"
        
        area = width * height
        rke = RipleysKEstimator(area = area, 
                                x_max = width, y_max = height,
                                y_min = 0, x_min = 0)
        
        """generate list of radii to assess. We generate 10 between 0 
        and the root of half the total area. More radii would give a higher 
        resolution but increases the computation time.
        
        see https://wiki.landscapetoolbox.org/doku.php/spatial_analysis_methods:
        ripley_s_k_and_pair_correlation_function
        for more details on this"
        """
        r = np.linspace(0, np.sqrt(area/2), 10)
        
        "generate the full list of radii for data frame later."
        "just repeats r above for how many models we have"
        rs = [r]*len(collisions)
        
        "placeholder list for RK estimates"
        rkes = []        
        for i, collision in enumerate(collisions):
            
            """estimate RK curve given model collisions and list of radii
            Note mode arguement here for how to deal with common edge effect problem.
            Choice doesnt seem to have much effect in this case.
            Either none or translation recommended.
            """
            
            #rkes.append(rke(collisions, radii=r, mode='none')) 
            rkes.append(rke(collision, radii=r, mode='translation'))
            #rkes.append(rke(collisions, radii=r, mode='ohser')) 
            #rkes.append(rke(collisions, radii=r, mode='var-width')) 
            #rkes.append(ke(collisions, radii=r, mode='ripley')) 
            
            "this can take a long time so here's a progess bar"
            
            print("\r" + str((i+1)/len(collisions)*100) 
                  + "% complete                  ", end = "")

        
        
        return rkes, rs
       
    
    def panel_Regression_Prep(self, rkes, rs, id_number):
        
        """ Build the list of model RKEs into a dataframe
        
        Output dataframe has 4 columns
        
        x: list of radii on which RK is estimated. Renamed to x for use 
        in R regression later
        
        y: RKE curve corresponding to a given model's collisions and radii. 
        Renamed to y for R regression later
        
        ids: identifies which specific model the row belongs to.
        This has format "A_B" it is the Bth model of batch A. 
        For example if the id reads "0_8" it is from the eighth model of the 
        0th batch
        
        split: identifies which batch the row belongs to. Either 0 (control group)
        or 1 (test group)
        
        Parameters
        ------
        
        rkes, rs : list
            list of lists of radii `rs` and corresponding RK estimates `rkes`. 
            Each pair of sublists corresponds to an individual model.
            
        id_number : int
            Assign each row of the data frame a number 0 if it is a control group
            (e.g. python station results) or 1 if it is a test group (e.g. cpp results.)
            
        Returns
        ------
        
        data : array_like
            assembled RK `data` from our list of models ready for fitting a
            regression in R. Should have 4 columns as defined above. 
        """
        
        "preprocessing"
        num_rs = len(rs[0])
        rkes = np.ravel(rkes)
        rs = np.ravel(rs)
        data = pd.DataFrame([rs, rkes]).T
        
        "rename rs and rkes columns for regression later."
        data.columns = ["x", "y"]
        
        """generate individual model ID numbers. Each model has 10 entries 
        (10 radii) so every 10 entrires belong to the ith model. Start
        with a list where every 10 values is one intiger (the model id).
        """
        
        ids = (np.arange(len(rkes))//num_rs).tolist()
        
        """ Then prefix every id with the batch number e.g. model 8 from 
        batch 0 becomes 0_8 . Allows unique IDs for every model such 
        that panel regression in R can recognise "individuals".
        """
        
        for i, item in enumerate(ids):
            ids[i] = str(id_number) + "_" + str(item)
            
        data["ids"] = ids
        
        """add a column with just the batch number. for batch 0 this 
        is a column of 0s.
        0 - control group. 1 - test group"""
        
        split = [id_number] * len(rkes)
        data["split"] = split
        
        return data
        
    
    def generate_Control_Frame(self, collisions, width, height):
        
        """generate a control group of RK curves to compare against
        
        - calculate RK curves of each model's collisions
        - store values in data frame.
        - save data frame as a control to load against in main
        
        Parameters
        ------
        collisions : list
            list of finished stationsim models `collisions` between agents.
        
        Returns
        ------
        data: array_like
            assembled RK `data` from our list of collisions ready for fitting a
            regression in R. Should have 4 columns as defined above in
            panel_regression_prep. 
        """
        
        rkes, rs = self.ripleysKE(collisions)
        data = self.panel_Regression_Prep(rkes, rs, 0)
        
        width = model_params["width"]
        height = model_params["height"]
        pop_total = model_params["pop_total"]
        gates_speed = model_params["gates_speed"]
               
        f_name = "RK_csvs/control_" + f"{width}_{height}_{pop_total}_{gates_speed}"
        f_name += ".csv"
        
        self.save_Frame(data, f_name)
        
        return data
        
    
    def generate_Test_Frame(self, collisions, width, height):
        
        """ Generate frame of test values to compare against
        
        Parameters
        ------
        collisions : list
            list of finished stationsim models `collisions` between agents.
        
        Returns
        ------
        data: array_like
            assembled RK `data` from our list of collisions ready for fitting a
            regression in R. Should have 4 columns as defined above in
            panel_regression_prep. 
        """
        rkes, rs = self.ripleysKE(collisions, width, height)
        data = self.panel_Regression_Prep(rkes, rs, 1)
        
        return data
        
    
    def save_Frame(self, data, f_name):
        
        """ Save a pandas data frame
        
        Parameters
        ------
        data : array_like
        
            pandas `data` frame. usually from generate_Control/Test_Frame
            output
        
        f_name : str
            `f_name` file name
            
        
        """
        
        data.to_csv(f_name, index = False)


    def load_Frame(self, f_name):
        
        """ Load a pandas data frame.
        
        Parameters
        ------        
        f_name : str
            `f_name` file name
            
        Returns
        ------
        data : array_like
        
            Pandas `data` frame usually from generate_Control/Test_Frame
            output.
        """
        
        return pd.read_csv(f_name)
    
    
    def collisions_kde(self, collisions, width, height):
        
        """ Plot spread of collisions through stationsim as a KDE plot
        
        Parameters
        ------
        collisions : list
            some list of coordinates where `collisions` occur
        """
        
        
        x = collisions[:,0]
        y = collisions[:,1]
        
        plt.figure()
        im = kdeplot(x, y)
        plt.xlim([0, width])
        plt.ylim([0, height])
        plt.xlabel("StationSim Width")
        plt.ylabel("StationSim Height")
        plt.title("KDE of Agent Collisions over StationSim")
    
    
    def spaghetti_RK_Plot(self, data):
        
        """plot RK trajectories for several models and control and test batches

        Parameters
        ------
        
        data : array_like
            data frame from generate_Control/Test_Frame output
            
        """
        
        colours = ["black", "orangered"]
        "0 black for control models"
        "1 orange for test models"
        
        f = plt.figure()
        for item in set(data["ids"]):
            sub_data = data.loc[data["ids"] == item]
            
            rs = sub_data["x"]
            rkes = sub_data["y"]
            split = (sub_data["split"]).values[0]
            plt.plot(rs,rkes, color = colours[split])
            
        plt.plot(-1, -1, alpha = 1, color = colours[0], 
                 label = "Control Group RKs")
        plt.plot(-1, -1, alpha = 1, color = colours[1], 
                 label = "Test Group RKs")
        plt.xlabel("radius of RK circle")
        plt.ylabel("RK Score")
        plt.legend()
 
    
    def notebook_RK_Plot(self, data1, data2):
        
        """Plot for RK notebook showing two extreme examples of RK curves

        The idea is to have two frames with exactly one model in that show two
        extreme cases of collisions clustering. We have a tight clustering case
        in orange with a rapidly increasing RK score and a sparse case with a 
        shallow linear RK score.

        Parameters
        ------

        data1, data2 : array_like
            `data1` and `data2` are two RKE dataframes from generate_Control_Frame.
            They have a structure specified in said function.

        """
        colours = ["black", "orangered"]
        "0 black for control models"
        "1 orange for test models"
        
        f = plt.figure()
       
        rs1 = data1["x"]
        rkes1 = data1["y"]
        plt.plot(rs1,rkes1, color = colours[0], label = "Dispersed Queueing Case")
         
        rs2 = data2["x"]
        rkes2 = data2["y"]
        plt.plot(rs2,rkes2, color = colours[1], label = "Clustered Queueing Case")
        
        plt.xlabel("radius of RK circle")
        plt.ylabel("RK Score")
        plt.legend()
        
        
    def main(self, test_collisions, model_params):
     
        """Main function for comparing python and cpp outputs.
        
        - Check a control file exists given specified model parameters
        - If it exists, load it. If not, generate one using 100 model runs
        - Generate control group data frame
        - Calculate RK curves of test models.
        - Generate corresponding test group data frame.
        - concat control and test frames and save for analysis in R
        using RK_population_modelling.R
        
        Parameters
        ------
        
        test_collisions : list
            list of (nx2) array stationsim model 'collisions'
            
        Returns 
        ------
        
        data : array_like
            Joined pandas `data` frame for control and test groups.
        """
        
        "generate control file name to load from model parameters"
        width = model_params["width"]
        height = model_params["height"]
        pop_total = model_params["pop_total"]
        gates_speed = model_params["gates_speed"]
            
        f_name = "RK_csvs/control_" + f"{width}_{height}_{pop_total}_{gates_speed}" + ".csv"
        
        "try loading said file. if no file make and save one."
        try:
            data_control = self.load_Frame(f_name)
            print("Control data found at: " + f_name)
        except:
            print("No control frame found for given parameters.")
            print("Generating control frame using large number of models (100).")
            print("This may take a while if you have a large population of agents")
            control_models = self.generate_Model_Sample(100, model_params)
            data_control = self.generate_Control_Frame(control_models, width, height)
        
        "generate data frame from test_collisions"
        data_test = self.generate_Test_Frame(test_collisions, width, height)
        
        "join control and test ferames and save as joined frame."
        data = pd.concat([data_control, data_test])
        f_name = "RK_csvs/joint_" + f"{width}_{height}_{pop_total}_{gates_speed}"
        f_name += ".csv"
        self.save_Frame(data, f_name)

        return data
    
  
def collision_Folder_Name(source, parameters):
    
    """ name of subfolder to generate sub csvs into
    
    folder name consists of three parts
    source - where do collisions come from
    parameters - list of important stationsim parameters saved into file name
    unique id - identifier for multiple runs of same parameters. uses string of 
    time from datetime.strftime. This should be universally unique. Could also
    used something like uuid4 for easier crossplatform file generation if you want
    to save csvs in cpp as well.
    
    
    Parameters
    ------
    source : str
        which language do collisions come from. python cpp etc.
        
    parameters : list
        List of important parameters from stationsim to load in file name.
        We use model width, height, pop_total, and gates_speed.
    
    Returns
    ------
    folder_name : string
        `folder_name` name of folder to save files to.
        
    """
    parameter_string = ""
    for item in parameters:
        parameter_string += str(item) + '_'
    
    "uuid provides unique indentifier in string."
    time_id =  datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    folder_name = source + "_" + parameter_string + time_id
    return folder_name
    

def save_Collision_csvs(collisions, folder_name):
    
    """ save model collision csvs
    
    Parameters
    ------
    collisions : list 
        list of (nx2) numpy `collisions` arrays.
        
    folder_name : string
        `folder_name` Name of folder csvs are saved to.
    """
        
    os.mkdir(folder_name)
    for i, collision in enumerate(collisions):
        file_name = folder_name + "/" + str(i) + ".csv"
        frame = pd.DataFrame(collision)
        frame.to_csv(file_name, index = False)
    
    
def load_Collision_csvs(folder_name):
    
    """ load model collision csvs
    
    Parameters
    ------
    folder_name : string
        `folder_name` Name of folder csvs are saved to.
        
    Returns
    ------
    collisions : list 
        list of (nx2) numpy `collisions` arrays.  
    """
    
    collisions = []
    files = glob.glob(folder_name + "/*")
    for file in files:
        data = pd.read_csv(file)
        collisions.append(np.array(data))
    
    return collisions

#%%
if __name__ == "__main__":
    
    "specify stationsim params"
    model_params = {

    'width': 200,
    'height': 50,
    'pop_total': 30,
    'gates_speed': 1,

    'gates_in': 3,
    'gates_out': 2,
    'gates_space': 1,
    
    'speed_min': .2,
    'speed_mean': 1,
    'speed_std': 1,
    'speed_steps': 3,
    
    'separation': 5,
    'max_wiggle': 1,
    
    'step_limit': 3600,
    
    'do_history': True,
    'do_print': True,
    
    "random_seed" : None,
    }
    """number of model repetitions n. Recommend at least 30 for <50 agents
     and at least 100 for >50 agents."""
     
    n_test_runs = 10
    "init"
    ssRK = stationsim_RipleysK()
    "generate models to test. done in python here but can swap as necessary."
    "could even reduce this to just each model's collisions"
    test_models = ssRK.generate_Model_Sample(n_test_runs, model_params)
    "convert to collisions"
    test_collisions = [np.vstack(model.history_collision_locs) for model in test_models]
    
    data = ssRK.main(test_collisions, model_params)
    ssRK.spaghetti_RK_Plot(data)
    
