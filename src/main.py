import numpy as np
import pandas as pd
from kmeans import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def match_labels(labels, truth):
    matched = np.zeros_like(labels)
    # print(truth)
    # print(labels)
    # print(np.unique(truth))
    # print(np.unique(labels))

    # match_count = np.zeros_like(np.unique(truth))
    # print(match_count)
    for l in np.unique(labels):

        # get array of size [1 ... t] where each position is the count of times label l corresponded to value t
        match_count = [np.sum((labels==l)*(truth==t)) for t in np.unique(truth)]
        print(match_count)
        matched[labels==l] = np.unique(truth)[np.argmax(match_count)]
        # print(matched)
    return matched
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

# Drop the first two columns and make arrays only with gene attributes 
cho_arr = cho_df.drop(columns=[0,1]).to_numpy()
iyer_arr = iyer_df.drop(columns=[0,1]).to_numpy()


np.random.seed(17)

cho_kmeans = Kmeans(cho_arr)
cho_labels = cho_kmeans.run()

# get ground truth values
cho_truth = cho_df[1].to_numpy()

# match labels with ground truth values
cho_labels_match = match_labels(cho_labels, cho_truth)

# Compute confusion matrix
cm = confusion_matrix(cho_truth, cho_labels_match)

# Plot confusion matrix
plt.imshow(cm,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, z, ha='center', va='center')
plt.xlabel("kmeans label")
plt.ylabel("truth label")
plt.show()

# np.random.seed(2)

# iyer_kmeans = Kmeans(iyer_arr, k=10)
# iyer_labels = iyer_kmeans.run()

# # get ground truth values
# iyer_truth = iyer_df[1].to_numpy()

# # match labels with ground truth values
# iyer_labels_match = match_labels(iyer_labels, iyer_truth)

# print(iyer_labels_match - iyer_truth)
# iyer_kmeans = Kmeans(iyer_arr, 10)
# iyer_kmeans.info()

