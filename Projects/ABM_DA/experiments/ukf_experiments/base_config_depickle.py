import glob
import numpy as np
import pickle
import pandas as pd

n=5
rates = [1,2,5,10,20,50,100] #.2 to 1 by .2
noises = [0,0.25,0.5,1,2,5,10,25,50,100]
run_id = np.arange(0,5,1) #20 runs


errors = []
rates2 = []
noises2 = []
run_id2 = []

 
for i in rates:
    for j in noise:
        for k in run_id:
            file = glob.glob(f"ukf_results/agents_{n}_rate_{i}_noise_{j}_base_config_errors_{k}")
            if file != []:
                file=file[0]
                rates2.append(i)
                noises2.append(j)
                run_id2.append(k)
                
                f = open(file,"rb")
                error = pickle.load(f)
                f.close()
                errors.append(error)



run_id2 = np.vstack(run_id2)
rates2 = np.vstack(rates2)
noises2 = np.vstack(noises2)
errors2 = np.vstack(errors)

data = pd.DataFrame(np.concatenate((run_id2,rates2,noises2,errors2),axis=1))
data.columns = ["run_id","rates","noise","actual","prediction","ukf"]
data2 = data.groupby(["rates","noise"],as_index=False).agg("mean")

maxes = []
for i in range(errors2.shape[0]):
    row = np.array([data2["actual"][i],data2["prediction"][i],data2["ukf"][i]])
    maxes.append(np.where(np.max(row))[0][0])

data2["maxes"] = maxes


rates = [str(rate) for rate in rates]
noises = [str(noise) for noise in noises]

max_array = np.ones((len(rates),len(noises)))*np.nan
#rows are rates,columns are noise
for i,rate in enumerate(rates):
    for j,noise in enumerate(noises):
        rate_rows = data2.loc[data2["rates"]==rate]
        rate_noise_row = rate_rows.loc[rate_rows["noise"]==noise]
        max_array[i,j]= rate_noise_row["maxes"][0]
        
"discrete matrix with y labels rates x labels noises  for imshow
max_array = np.flip(max_array,axis=0)

f,ax = plt.subplots()
ax.imshow(max_array,origin="lower")
ax.set_xticks(np.arange(len(noises)))
ax.set_yticks(np.arange(len(rates)))
ax.set_xticklabels(noises)
ax.set_yticklabels(rates)
plt.xlabel("noise")
plt.ylabel("sampling rate")
plt.title("base configuration experiment varying rate and noise to find best performer of obs preds and ukf")