import pandas as pd
import os
import numpy as np
import random

from numpy import random
#
data_folder = r'rain_data'

# Create a list of all the csv files in the data folder
csv_files = [os.path.join(data_folder, x) for x in os.listdir(data_folder) if x.endswith('.csv')]
# Create a list of all the csv files in the data folder

#randomly select 80% of the files for training and 20% for testing

train_files = random.choice(csv_files, int(len(csv_files)*0.8), replace=False)

test_files = [x for x in csv_files if x not in train_files]

# concatenate all the training files into a single dataframe
train_df = pd.concat((pd.read_csv(f) for f in train_files))


# concatenate all the testing files into a single dataframe
test_df = pd.concat((pd.read_csv(f) for f in test_files))

# save the training and testing dataframes to csv files
train_df.to_csv('data_for_model//train_data.csv', index=False)
test_df.to_csv('data_for_model//test_data.csv', index=False)