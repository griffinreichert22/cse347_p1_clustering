import numpy as np
import pandas as pd
from clustering import *

## Get user input from command line
# print("Welcome to my clustering alg!")
# print("  [1] K-means clustering")
# print("  [2] Spectral clustering")
# inpt = input("Please enter the number of the algorithm you wish to use: ")
# while inpt != "1" and inpt != "2":
#     inpt = input("Whoops! Please only enter '1' or '2' ('q' to quit): ")
#     if inpt == "q":
#         break

inpt = "1" # REMOVE ME
## Read in data from text files

# define file paths to /data directory
data_fp = "../data/"
cho_fp = data_fp + "cho.txt"
iyer_fp = data_fp + "iyer.txt"

# Read data from specified file paths into pandas dataframes
cho_df = pd.read_csv(cho_fp, header=None, sep='\t')
iyer_df = pd.read_csv(iyer_fp, header=None, sep='\t')

# remove outlier values (denoted as -1 in colum 1)
cho_df = cho_df[cho_df[1] > 0]
iyer_df = iyer_df[iyer_df[1] > 0]

# Drop the first two and make arrays only with gene attributes 
cho_arr = cho_df.drop(columns=[0,1]).to_numpy()
iyer_arr = iyer_df.drop(columns=[0,1]).to_numpy()

print(cho_arr)
print(iyer_arr)

alg = Kmeans(iyer_arr)
# alg.info()
# elif inpt == "2":
#     spectral(cho_data)