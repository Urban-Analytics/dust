
from ukf_plots import plots
import numpy as np 
import pickle

pop_total = 10
a = np.load(f"actual_{pop_total}.npy")
b = np.load(f"ukf_{pop_total}.npy")

f = open(f"pickle_model_ukf_{pop_total}","rb")
U = pickle.load(f)
f.close()
        
plots.diagnostic_plots(U,a,b,False,True)
plots.diagnostic_plots(U,a,b,True,True)
