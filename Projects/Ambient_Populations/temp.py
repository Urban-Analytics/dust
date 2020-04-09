gro#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:40:56 2020

@author: nick
"""

import pandas as pd
import os
import numpy as np


#%%
data_dir = "./data/lcc_footfall" # Where to save the csv files


# Connect to the Data Mill North page and parse the html
root = 'https://datamillnorth.org/dataset/leeds-city-centre-footfall-data'
soup = BeautifulSoup(urlopen(root), 'html.parser')

# Iterate over all links and see which are csv files
for link in soup.find_all('a'):
    #print("\n****",link,"****\n")
    url = link.get('href')
    if url==None: # if no 'href' tag
        continue
    
    if url.endswith(".csv"):
        filename = url.strip().split("/")[-1] # File is last part of the url
        
        # For some reason some files are duplicated
        if filename.startswith("Copy") or filename.startswith("copy"): 
            continue
        # And we don't care about xmas analysis
        if filename.startswith("Christ"):
            continue
        
        # Save the csv file (unless it already exists already)
        full_path = os.path.join("./data/lcc_footfall",filename)
        if os.path.isfile(full_path):
            print("File {} exists already, not downloading".format(filename))
        else:
            print("Downloading {}".format(filename)) 
            csv_url = "https://datamillnorth.org/"+url
            data = pd.read_csv(csv_url)
            data.to_csv(full_path)

#%% Make a big data frame

def convert_hour(series):
    """Assumes the given series represents hours. Works out if they're 
    integers or in the format '03:00:00' and returns them as integers"""
    
    # If it's a number then just return it
    if isinstance(series.values[0], np.int64) or isinstance(series.values[0], np.float64) or isinstance(series.values[0], float):
        return series
    
    # If it's a string see if it can be made into a number
    try:
        int(series.values[0])
        return pd.to_numeric(series)
    except: # If get here then it couldn't be made into an integer
        pass
    
    if ":" in series.values[0]:
        return pd.to_numeric(series.apply(lambda x: x.strip().split(":")[0]))
    
    # If here then I don't know what to do.
    raise Exception("Unrecognised type of hours: {}".format(series))
    
# Template for our data frame. Set the type as well (default is OK for 'location')
template = pd.DataFrame(columns = ["Location", "Date", "Hour", "Count", "DateTime"])
template["Date"] = pd.to_datetime(template["Date"])
template["Hour"] = pd.to_numeric(template["Hour"])
template["Count"] = pd.to_numeric(template["Count"])
template["DateTime"] = pd.to_numeric(template["DateTime"]) # (this one is derived from date and hour)

frames = [] # Build up a load of dataframes then merge them
total_rows = 0 # For checking that the merge works
files = [] # Remember the names of the files we tried to analyse
failures= [] # Remember which ones didn't work


# Read the files in
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        try:
            #print(filename)
            files.append(filename)
            df = pd.read_csv(os.path.join(data_dir,filename))
            
            # Check the file has the columns that we need, and work out what the column names are for this file (annoyingly it changes)
            date_col = "Date" # Doesn't change
            count_col = "Count" if "Count" in df.columns else "InCount" # Two options
            hour_col = "Hour" 
            loc_col = "Location" if "Location" in df.columns else "LocationName"
            
            if False in [date_col in df.columns, count_col in df.columns, hour_col in df.columns, loc_col in df.columns]:
                raise Exception("File '{}' is missing a column. Date? {}, Count? {}, Hour? {}, Location? {}".
                      format(filename, date_col in df.columns, count_col in df.columns, hour_col in df.columns, loc_col in df.columns))
                

            # Check if any of the columns have nans
            bad_cols = []
            for x in [date_col, count_col, hour_col, loc_col]:
                if True in df[x].isnull().values:
                   bad_cols.append(x)
            if len(bad_cols)>0:
                failures.append(filename)
                print(f"File {filename} has nans in the following columns: '{str(bad_cols)}'. Ignoring it")
                continue

            
            # Create Series' that will represent each column
            dates  = pd.to_datetime(df[date_col])
            counts = pd.to_numeric(df[count_col])
            hours  = convert_hour(df[hour_col]) # Hours can come in different forms 
            locs   = df[loc_col]
            
            # Derive a proper date from the date and hour
            # (Almost certainly a more efficient way to do this using 'apply' or whatever)
            dt     = pd.to_datetime(pd.Series( data = [date.replace(hour=hour) for date,hour in zip(dates,hours) ] ) )
            
            #df.apply(lambda x: x[date_col].replace(hour = x[hour_col]), axis=1)
            
            if False in [len(df) == len(x) for x in [dates, counts, hours, locs, dt]]:
                raise Exception("One of the dataframe columns does not have enough values")
            total_rows += len(df)
                
            
            # Create a temporary dataframe to represent the information in that file.
            # Note that consistent column names (defined above) are used
            frames.append(pd.DataFrame(data={"Location":locs, "Date":dates, "Hour":hours, "Count":counts, "DateTime":dt}))
        except Exception as e:
            print("Caught exception on file {}".format(filename))
            raise e
            

# Finally megre the frames into one big one
merged_frames = pd.concat(frames)
if total_rows != len(merged_frames):
    raise Exception(f"The number of rows in the individual files {total_rows} does \
not match those in the final dataframe {len(merged_frames)}.")

df = template.append(merged_frames)            
print(f"Finished. Made a dataframe with {len(df)} rows. {len(failures)}/{len(files)} files could not be read.")
        

#%%


        
        