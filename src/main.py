import numpy as np
import pandas as pd
from kmeans import *
from spectral import *
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# sklearn clustering methods used for comparison
from sklearn.cluster import KMeans, SpectralClustering

np.set_printoptions(precision=3)

def match_labels(labels, truth):
    matched = np.zeros_like(labels)
    for l in np.unique(labels):
        # get array of size [1 ... t] where each position is the count of times label l corresponded to value t
        match_count = [np.sum((labels==l)*(truth==t)) for t in np.unique(truth)]
        matched[labels==l] = np.unique(truth)[np.argmax(match_count)]
    return matched

def internal_index(X, labels):
    # print("finding wss...")
    clusters = np.unique(labels)
    # print(clusters)
    # print(c_idx)
    wss = 0
    for c in clusters:
        c_idx = np.where(labels==c)[0]
        c_points = X[c_idx]

        # find the centroid
        centroid = np.mean(c_points, axis=0)
        for x in c_points:
            wss += calc_dist(x, centroid)**2
    return wss

def do_confusion_matrix(labels, truth, plot=False):
    # Compute confusion matrix
    cm = confusion_matrix(truth, labels)
    purity1 = np.sum(np.amax(cm, axis=0))
    purity2 = np.sum(cm)
    # print(f'num: {purity1} denom: {purity2}')
    # Plot confusion matrix
    if plot:
        plt.imshow(cm,interpolation='none',cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            plt.text(j, i, z, ha='center', va='center')
        plt.xlabel("predicted label")
        plt.ylabel("truth label")
        plt.show()
    return purity1/purity2


# Get user input from command line
print("Welcome to my clustering alg!")
print("  [1] K-means clustering")
print("  [2] Spectral clustering")
inpt = input("Please enter the number of the algorithm you wish to use: ")
while inpt != "1" and inpt != "2" and inpt != 'q':
    inpt = input("Whoops! Please only enter '1' or '2' ('q' to quit): ")

if inpt != 'q':
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



    # get ground truth values
    cho_truth = cho_df[1].to_numpy()
    iyer_truth = iyer_df[1].to_numpy()

    cho_pred = []
    # cho_sklearn = []
    iyer_pred = []
    # iyer_sklearn = []
    # get kmeans labels
    if inpt == '1':
        print("\nK-MEANS CLUSTERING")
        np.random.seed(23)
        cho_kmeans = Kmeans(cho_arr)
        cho_pred = cho_kmeans.run()
        # cho_sklearn = KMeans(n_clusters=5, random_state=0).fit(cho_arr).labels_
        
        np.random.seed(22)
        iyer_kmeans = Kmeans(iyer_arr, k=10)
        iyer_pred = iyer_kmeans.run()
        # iyer_sklearn = KMeans(n_clusters=10, random_state=0).fit(iyer_arr).labels_

    # get spectral labels
    elif inpt == '2':
        print("\nSPECTRAL CLUSTERING")
        np.random.seed(49)
        cho_spec = Spectral(cho_arr, 49)
        cho_pred = cho_spec.run()
        # cho_sklearn = SpectralClustering(n_clusters=5, random_state=0).fit(cho_arr).labels_
        
        np.random.seed(22)
        iyer_spec = Spectral(iyer_arr, 90)
        iyer_pred = iyer_spec.run()
        # iyer_sklearn = SpectralClustering(n_clusters=10, random_state=0).fit(iyer_arr).labels_

    cho_matched = match_labels(cho_pred, cho_truth)
    # cho_sklearn = match_labels(cho_sklearn, cho_truth)
    cho_pred_wss = internal_index(cho_arr, cho_pred)
    cho_truth_wss = internal_index(cho_arr, cho_truth)
    # cho_sklearn_wss = internal_index(cho_arr, cho_sklearn)
    cho_purity = do_confusion_matrix(cho_matched, cho_truth)
    # cho_sklearn_purity = do_confusion_matrix(cho_sklearn, cho_truth)
    # # Uncomment to plot confusion matrix
    # cho_purity = do_confusion_matrix(cho_matched, cho_truth, True)
    print("\n___ CHO ___")
    print("Internal Index:")
    print(f" pred wss:    {cho_pred_wss:.3f}")
    print(f" truth wss:   {cho_truth_wss:.3f}")
    # print(f" sklearn wss: {cho_sklearn_wss:.3f}")

    print("External Index")
    print(f" pred purity:    {cho_purity:.5f}")
    # print(f" sklearn purity: {cho_sklearn_purity:.5f}")
    print("Labels:")
    print(cho_matched)


    iyer_matched = match_labels(iyer_pred, iyer_truth)
    # iyer_sklearn = match_labels(iyer_sklearn, iyer_truth)
    iyer_pred_wss = internal_index(iyer_arr, iyer_pred)
    iyer_truth_wss = internal_index(iyer_arr, iyer_truth)
    # iyer_sklearn_wss = internal_index(iyer_arr, iyer_sklearn)
    iyer_purity = do_confusion_matrix(iyer_matched, iyer_truth)
    # iyer_sklearn_purity = do_confusion_matrix(iyer_sklearn, iyer_truth)
    # # Uncomment to plot confusion matrix
    # iyer_purity = do_confusion_matrix(iyer_matched, iyer_truth, True)
    print("\n__ IYER __")
    print("Internal Index:")
    print(f" pred wss:    {iyer_pred_wss:.3f}")
    print(f" truth wss:   {iyer_truth_wss:.3f}")
    # print(f" sklearn wss: {iyer_sklearn_wss:.3f}")
    print("External Index")
    print(f" pred purity:    {iyer_purity:.5f}")
    # print(f" sklearn purity: {iyer_sklearn_purity:.5f}")
    print("Labels:")
    print(iyer_matched)