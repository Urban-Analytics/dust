import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone, utc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

au_tz = timezone('Australia/Sydney')
fn = "Transurban_1516171819_data.pkl"
fn2 = "SCATS_2019_sumdata.pkl"

def scats_imports(savedata=True, filen=fn):
    """Data prep from original files: scats.
    The raw data filenames should be the default download names,
    e.g. scats_1_2018.csv"""
    m2counts18=pd.DataFrame()
    for j in range(12):
        ju = j+1
        m2counts18=m2counts18.append(pd.read_csv("./scats/scats/scats_"+str(ju)+"_2019.csv"))


    #Restrict the DataFrame to the features of interest
    #Fill empty entries with 0 (no counts detected), the 'sum' aggregation is just a sum over a single entry (no effect)
    #m2counts18_r = m2counts18[['SDateTime','VehicleClass','TollPointID','TotalVolume']].pivot_table(index='SDateTime', columns=['TollPointID','VehicleClass'], values='TotalVolume', aggfunc='sum', fill_value=0.0)
    #counts18_r = m2counts18_r.join(lccounts18_r)
    #print(counts18_r.iloc[:2])

    if savedata:
        m2counts18.to_pickle(fn2)

    return m2counts18

def transurban_imports(savedata=True, filen=fn):
    """Data prep from original files: M2 and LCT counts.
    The raw data filenames should be the default download names,
    e.g. M2_trips_2018-01.csv"""
    print("Hello")
    m2counts18=pd.DataFrame()
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        m2counts18=m2counts18.append(pd.read_csv("./Transurban2020data/M2_trips_2015-"+k+".csv"))
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        m2counts18=m2counts18.append(pd.read_csv("./Transurban2020data/M2_trips_2016-"+k+".csv"))
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        m2counts18=m2counts18.append(pd.read_csv("./Transurban2020data/M2_trips_2017-"+k+".csv"))
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        m2counts18=m2counts18.append(pd.read_csv("./Transurban2020data/M2_trips_2018-"+k+".csv"))
    for j in range(9):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        m2counts18=m2counts18.append(pd.read_csv("./Transurban2020data/M2_trips_2019-"+k+".csv"))

    lccounts18=pd.DataFrame()
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        lccounts18=lccounts18.append(pd.read_csv("./Transurban2020data/LCT_trips_2015-"+k+".csv"))
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        lccounts18=lccounts18.append(pd.read_csv("./Transurban2020data/LCT_trips_2016-"+k+".csv"))
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        lccounts18=lccounts18.append(pd.read_csv("./Transurban2020data/LCT_trips_2017-"+k+".csv"))
    for j in range(12):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        lccounts18=lccounts18.append(pd.read_csv("./Transurban2020data/LCT_trips_2018-"+k+".csv"))
    for j in range(9):
        ju = j+1
        if len(str(ju)) == 1:
            k = '0'+str(ju)
        else:
            k = str(ju)
        lccounts18=lccounts18.append(pd.read_csv("./Transurban2020data/LCT_trips_2019-"+k+".csv"))

    #Combine date and time fields to DateTime object
    #This is defined in the UTC timezone to avoid local timezone daylight saving overlaps
    datetimel=[]
    for ind,j in m2counts18.iterrows():
        date0=j['Date']
        time0=j['IntervalStart']
        datespl0=date0.split('-')
        timespl0=time0.split(':')
        (y0, m0, d0) = datespl0
        (h0, min0) = timespl0
        dt=au_tz.localize(datetime(int(y0), int(m0), int(d0), int(h0), int(min0), 0)).astimezone(utc)
        datetimel.append(dt)
    m2counts18['SDateTime']=datetimel #Add 'interval start' datetime column

    datetimel2=[]
    for ind,j in lccounts18.iterrows():
        date0=j['Date']
        time0=j['IntervalStart']
        datespl0=date0.split('-')
        timespl0=time0.split(':')
        (y0, m0, d0) = datespl0
        (h0, min0) = timespl0
        dt=au_tz.localize(datetime(int(y0), int(m0), int(d0), int(h0), int(min0), 0)).astimezone(utc)
        datetimel2.append(dt)
    lccounts18['SDateTime']=datetimel2


    #For unique feature identifiers, the following four are sufficient (see the Data_Dictionary file to validate this)
    #SDateTime, VehicleClass, TollPointID, TotalVolume
    #Note that I checked the 'Version' was consistent across the data
    #GantryType could add extra useful info... leave it for now, this feature can be learned from TollPointID.

    #Restrict the DataFrame to the features of interest
    #Fill empty entries with 0 (no counts detected), the 'sum' aggregation is just a sum over a single entry (no effect)
    m2counts18_r = m2counts18[['SDateTime','VehicleClass','TollPointID','TotalVolume']].pivot_table(index='SDateTime', columns=['TollPointID','VehicleClass'], values='TotalVolume', aggfunc='sum', fill_value=0.0)
    lccounts18_r = lccounts18[['SDateTime','VehicleClass','TollPointID','TotalVolume']].pivot_table(index='SDateTime', columns=['TollPointID','VehicleClass'], values='TotalVolume', aggfunc='sum', fill_value=0.0)
    counts18_r = m2counts18_r.join(lccounts18_r)
    print(counts18_r.iloc[:2])

    if savedata:
        counts18_r.to_pickle(fn)

    return counts18_r

def transurban_load(filen=fn):
    """Load if the data has previously been imported and saved as a pickle file."""
    df = pd.read_pickle(fn)
    print(df.iloc[:2])
    return df

def daylight_fix(df):
    df_daylightfix=df.iloc[8648:8652]
    df_daylightfix.index = df_daylightfix.index - timedelta(minutes=60)
    df2 = pd.concat([df.iloc[:8648],df_daylightfix,df.iloc[8648:]])
    return df2

def scats_load(filen=fn2):
    sc = pd.read_pickle(fn2)
    print(sc.iloc[:2])
    return sc

def inci_load():
    inci=pd.read_csv("inlast_update_feb18major.csv")
    incit = pd.read_csv("inlast_update_mar18major.csv")
    inci=inci.append(incit,ignore_index=True)
    incit = pd.read_csv("inlast_update_apr18major.csv")
    inci=inci.append(incit,ignore_index=True)
    print('Number of incidents: '+str(len(inci)))
    return inci

def inci_align(inci, df, sc, vis=False):
    #List of incident reported start times
    idatetimel=[]
    for j in enumerate(inci['ns2:start-time']):
        date0 = j[1].split('T')[0]
        time0 = (j[1].split('T')[1]).split('+')[0]
        (y0, m0, d0) = date0.split('-')
        (h0, min0, sec0) = time0.split(':')
        dt=au_tz.localize(datetime(int(y0), int(m0), int(d0), int(h0), int(min0), int(float(sec0))))#.astimezone(utc)
        dt = dt.astimezone(utc)
        idatetimel.append(dt)

    #List of Transurban (and SCATS) indices giving the next count after the incident start.
    ind_istart=[]
    for istart in idatetimel:
        for ind,j in enumerate(df.index):
            if j>istart:
                ind_istart.append(ind)
                break
            else:
                continue
    
    if vis: #Visualise a few incidents and the resulting time series data
        il = ([27, 451])
        #week = 4*24*7
        for j in il:
            dur=inci['duration'][j]/15
            plt.clf()
            for ind, k in enumerate(sc.iloc[ind_istart[j]-4:ind_istart[j]+4].values.T):
                plt.plot(k,color='C'+str(ind % 10))
                #plt.plot(sc.iloc[ind_istart[j]-4+week:ind_istart[j]+4+week].values.T[ind],'--',color='C'+str(ind % 10),alpha=0.3)
                #plt.plot(sc.iloc[ind_istart[j]-4+2*week:ind_istart[j]+4+2*week].values.T[ind],'--',color='C'+str(ind % 10),alpha=0.3)
            
            plt.plot([4,4],[0,np.max(sc.iloc[ind_istart[j]-4:ind_istart[j]+4].values)])
            if dur<6:
                plt.plot([4+dur,4+dur],[0,np.max(sc.iloc[ind_istart[j]-4:ind_istart[j]+4].values)])
            else:
                plt.plot([4+6,4+6],[0,np.max(sc.iloc[ind_istart[j]-4:ind_istart[j]+4].values)])
            plt.xlabel('Timesteps (15 mins)')
            plt.ylabel('Counts')
            plt.title('SCATS detectors')
            plt.show()

            plt.clf()
            for ind, k in enumerate(df.iloc[ind_istart[j]-4:ind_istart[j]+4].values.T):
                plt.plot(k,color='C'+str(ind % 10))
                #plt.plot(df.iloc[ind_istart[j]-4+week:ind_istart[j]+4+week].values.T[ind],'--',color='C'+str(ind % 10),alpha=0.3)
                #plt.plot(df.iloc[ind_istart[j]-4+2*week:ind_istart[j]+4+2*week].values.T[ind],'--',color='C'+str(ind % 10),alpha=0.3)
            
            plt.plot([4,4],[0,np.max(df.iloc[ind_istart[j]-4:ind_istart[j]+4].values)])
            if dur<6:
                plt.plot([4+dur,4+dur],[0,np.max(df.iloc[ind_istart[j]-4:ind_istart[j]+4].values)])
            else:
                plt.plot([4+6,4+6],[0,np.max(df.iloc[ind_istart[j]-4:ind_istart[j]+4].values)])
            plt.xlabel('Timesteps (15 mins)')
            plt.ylabel('Counts')
            plt.title('Transurban detectors')
            plt.show()

    return ind_istart

# def aggvehicles():
#     """Combine truck and car features"""

def input_prep(data,train_from,train_to,n_test, predictsteps=2, lookbacksteps=8):
    train_dat = np.array(data[train_from:train_to])
    test_dat = np.array(data[train_to:][:n_test])
    #Scale on training data (fit and transform)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_dat = train_dat.astype('float32')
    train_scaled = scaler.fit_transform(train_dat)
    test_scaled = scaler.transform(test_dat)

    train_X=[]
    train_y=[]
    test_X=[]
    test_y=[]

    for j in range(len(train_scaled)-int(predictsteps+lookbacksteps-1)):
        train_datset=train_scaled[j:lookbacksteps+j,:]
        train_ycomp=train_scaled[int(predictsteps+lookbacksteps-1)+j]
        train_X.append(train_datset)
        train_y.append(train_ycomp)

    for j in range(len(test_scaled)-int(predictsteps+lookbacksteps-1)):
        test_datset=test_scaled[j:lookbacksteps+j,:]
        test_ycomp=test_scaled[int(predictsteps+lookbacksteps-1)+j]
        test_X.append(test_datset)
        test_y.append(test_ycomp)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)

    print('train_X shape: '+str(train_X.shape))
    return (train_X, train_y, test_X, test_y, scaler)

def plot_loss(history):
    plt.plot(history.history['loss'],label='Training loss')
    plt.plot(history.history['val_loss'],label='Test loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

def evaluate(dats, model, scaler, predictsteps, lookbacksteps, with_scats=True):
    train_X = dats[0]
    train_y = dats[1]
    test_X = dats[2]
    test_y = dats[3]

    if with_scats:
        test_X2 = dats[4]
        yhat, yhat2 = model.predict([test_X, test_X2])

    else:
        #ythat = model.predict(train_X)
        yhat = model.predict(test_X)

    # Rescale values
    #train_rescpred=scaler.inverse_transform(ythat)
    test_rescpred=scaler.inverse_transform(yhat)
    #train_rescref=scaler.inverse_transform(train_y)
    test_rescref=scaler.inverse_transform(test_y)

    # Naive prediction benchmark (using previous observed value)
    testnpred=np.array(test_X).transpose(1,0,2)[-1]
    testnpredc=scaler.inverse_transform(testnpred)

    ## Performance measures
    seg_mael=[] #MAE list over detectors
    # seg_masel=[] #MASE list over detectors - this should use the historical norm
    seg_nmael=[] #Naive MAE list over detectors 

    for j in range(train_X.shape[-1]):
        
        seg_mael.append(np.mean(np.abs(test_rescref.T[j]-test_rescpred.T[j]))) #Mean Absolute Error
        seg_nmael.append(np.mean(np.abs(test_rescref.T[j]-testnpredc.T[j]))) #Mean Absolute Error for naive prediction
        # if seg_nmael[-1] != 0:
        #     seg_masel.append(seg_mael[-1]/seg_nmael[-1]) #Ratio of the two: Mean Absolute Scaled Error
        # else:
        #     seg_masel.append(np.NaN)
    
    return (np.array(seg_mael), np.array(seg_nmael), test_rescpred, test_rescref)
