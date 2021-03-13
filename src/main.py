import numpy as np
import pandas as pd
from kmeans import *
from spectral import *
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import networkx as nx

# float_formatter = lambda x: "%.2f" % x
np.set_printoptions(precision=3)

def match_labels(labels, truth):
    matched = np.zeros_like(labels)
    # print(truth)
    # print(labels)
    # print(np.unique(truth))
    # print(np.unique(labels))

    # match_count = np.zeros_like(np.unique(truth))
    # print(match_count)
    # count_map = []
    # label_map = np.zeros_like(np.unique(labels))
    for l in np.unique(labels):
        # get array of size [1 ... t] where each position is the count of times label l corresponded to value t
        match_count = [np.sum((labels==l)*(truth==t)) for t in np.unique(truth)]
        # count_map.append(match_count)
        # print(matched)
        matched[labels==l] = np.unique(truth)[np.argmax(match_count)]
        # label_map[l] = np.unique(truth)[np.argmax(match_count)]
    if (np.unique(matched).size != np.unique(truth).size):
        print("missing a class")
    # print(label_map)
    # print(count_map)
    return matched

def plot_cm(labels, truth):
    # Compute confusion matrix
    cm = confusion_matrix(truth, labels)
    # Plot confusion matrix
    plt.imshow(cm,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha='center', va='center')
    plt.xlabel("predicted label")
    plt.ylabel("truth label")
    plt.show()


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

### CHO KMEANS ###
"""np.random.seed(17)
# np.random.seed(12)

cho_kmeans = Kmeans(cho_arr)
cho_labels = cho_kmeans.run()

# get ground truth values
cho_truth = cho_df[1].to_numpy()

# match labels with ground truth values
cho_labels_match = match_labels(cho_labels, cho_truth)
plot_cm(cho_labels, cho_truth)

print("original labels")
print(cho_labels)
print("matched labels")
print(cho_labels_match)"""

### CHO SPECTRAL ###
# cho_spec = Spectral(cho_arr)
X = np.array([
    [1, 3], 
    [2, 1], 
    [3, 2], 
    [7, 8], 
    [9, 8],
    [8, 7], 
    [13, 14],
    [14, 14], 
    [15, 16]
])
x_spec = Spectral(X)
# plt.scatter(X[:, 0], X[:, 1])
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()



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

