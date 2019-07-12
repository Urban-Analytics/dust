
import pickle
sys.path.append('../../stationsim')
sys.path.append('../..')
from stationsim.ukf import ukf_ss
from stationsim.stationsim_model import Model

f = open(f"ukf_agents_5_prop_1.0-0.csv","rb")
u = pickle.load(f)
f.close()