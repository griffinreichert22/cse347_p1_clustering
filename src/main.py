import numpy as np
import pandas as pd
from clustering import *

## Get user input from command line
# print("Welcome to my clustering alg!")
# print("  [1] K-means clustering")
# print("  [2] Spectral clustering")
# alg = input("Please enter the number of the algorithm you wish to use: ")
# while alg != "1" and alg != "2":
#     alg = input("Whoops! Please only enter '1' or '2' ('q' to quit): ")
#     if alg == "q":
#         break

alg = "1" # REMOVE ME
## Read in data from text files


print("reading in data")
# file paths
data_fp = "../data/"
cho_fp = data_fp + "cho.txt"
iyer_fp = data_fp + "iyer.txt"

# cho_data = np.loadtxt(cho_fp)
# iyer_data = np.loadtxt(iyer_fp)

cho_data = pd.read_csv(cho_fp, header=None, sep='\t')
iyer_data = pd.read_csv(cho_fp, header=None, sep='\t')

# print(iyer_data)

if alg == "1":
    kmeans(cho_data)
elif alg == "2":
    spectral(cho_data)