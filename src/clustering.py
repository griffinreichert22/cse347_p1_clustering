import numpy as np
import pandas as pd
import random

def distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

## note that np.linalg.norm(a-b) can be used for euclidean distance
class Kmeans:
    def __init__(self, data, k=5, max_iters=50):
        self.k = k
        self.data = data
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.k)]
    
    def info(self):
        print("Printing info for kmeans")
        print("k: " + str(self.k))
        print("clust:")
        print(self.clusters)




    # # K-means alg 
    # choose k data points as initial clusters
    # while stop criteria not met
    #     for each data point x
    #         compute the distance from x to each centroid
    #         assign x to the closest centroid
    #     recompute  the centroids using the current cluster memberships
    #     check if stopping criteria is met
    # #


def spectral(data):
    print("TODO: spectral")
    print(data)

