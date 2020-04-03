#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 14:40:56 2020

@author: nick
"""

import pandas as pd
import os


#%%


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

#%%

def convert_hour(series):
    """Assumes the given series represents hours. Works out if they're 
    integers or in the format '03:00:00' and returns them as integers"""
    if isinstance(series.values[0], numpy.int64) or isinstance(series.values[0], numpy.float64) or isinstance(series.values[0], float):
        return series
    elif ":" in series.values[0]:
        return series.apply(lambda x: x.strip().split(":")[0])
    else:
        raise Exception("Unrecognised type of hours: {}".format(series))
    
# Single data frame. Set the type as well (dfault is OK for 'location')
data = pd.DataFrame(columns = ["Location", "Date", "Hour", "Count"])
data["Date"] = pd.to_datetime(data["Date"])
data["Hour"] = pd.to_numeric(data["Hour"])
data["Count"] = pd.to_numeric(data["Count"])

frames = [] # Build up a load of dataframes then merge them
failures= [] # Remember which ones didn't work

# Read the files in
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        try:
            #print(filename)
            df = pd.read_csv(os.path.join(data_dir,filename))
            # Check the file has the columns that we need
            has_date = "Date" in df.columns
            has_count = "Count" in df.columns or "InCount" in df.columns
            has_hour = "Hour" in df.columns
            has_loc = "Location" in df.columns or "LocationName" in df.columns
            
            if False in [has_date, has_count, has_hour, has_loc]:
                raise Exception("File '{}' is missing a column. Date? {}, Count? {}, Hour? {}, Location? {}".
                      format(filename, has_date, has_count, has_hour, has_loc))     
            
            # Create Series' that will represent each column
            dates  = pd.to_datetime(df["Date"])
            counts = pd.to_numeric(df["Count"]) if "Count" in df.columns else df["InCount"]
            # Hours can come in different forms 
            hours  = convert_hour(df["Hour"]) 
            locs   = df["Location"] if "Location" in df.columns else df["LocationName"]
            
            if False in [len(df) == len(x) for x in [dates, counts, hours, locs]]:
                raise Exception("One of the dataframe columns does not have enough values")
            
            # XXXX HERE
            # CAN'T CHECK THIS HERE, IT NEEDS TO BE DONE ABOVE, BEFORE TRYING TON 
            # WORK OUT WHAT TO DO WITH THE HOURS COLUMN
                
            # Check if any of the columns have nans
            for x,y in [("Dates", dates), ("Counts",counts), ("Hours", hours), ("Locs",locs)]:
                if True in y.isnull():
                    print("File {} has nans in the '{}' column".format(filename, x))
                    failures.append(filename)
                    continue
            
            # Create a temporary dataframe to represent the information in that file
            frames.append(pd.DataFrame(data={"Location":locs, "Date":dates, "Hour":hours, "Count":counts}))
        except Exception as e:
            print("Caught exception on file {}".format(filename))
            raise e
            
        
        
        
        
        