import numpy as np
import pandas as pd
    
def calc_dist(a, b):
    return np.linalg.norm(a-b)

# # K-means alg 
# choose k data points as initial clusters
# while stop criteria not met
#     for each data point x
#         compute the distance from x to each centroid
#         assign x to the closest centroid
#     recompute  the centroids using the current cluster memberships
#     check if stopping criteria is met
# #
class Kmeans:
    # constructor
    def __init__(self, data, k=5, max_iters=50):
        self.k = k
        # data should be a numpy ndarray
        self.data = data
        self.max_iters = max_iters
        # initialize list of k empty lists to hold clusters
        # each cluster will be a list of indicies
        self.clusters = [[] for _ in range(self.k)]
        # store mean feature for each centroid
        self.centroids = []
        self.n_samples, self.n_features = self.data.shape
    
    # prints info to terminal
    def info(self):
        print("Printing info for kmeans")
        print("k: " + str(self.k))
        print("clust:")
        print(self.clusters)
        print("data")
        print(self.data)

    def run(self):
        print("running kmeans...")
        # initialize centroids as random data points
        init_idx = np.random.choice(self.n_samples, self.k, replace=False)
        self.centroids = np.stack([self.data[i] for i in init_idx], axis=0)
        # cap runtime after the max number of iterations
        for it in range(self.max_iters):
            # print("iter "+ str(it))
            new_clusters = [[] for _ in range(self.k)]
            # iterate over all data points
            # i: index of each point, sample: value of each point
            for i, sample in enumerate(self.data):
                # compute distance from point to all centroids
                distances = [calc_dist(sample, c) for c in self.centroids]
                # find index of nearest centroid
                nearest_c_i = np.argmin(distances)
                # assign point to cluster of nearest centroid
                new_clusters[nearest_c_i].append(i)
            # update clusters
            self.clusters = new_clusters

            # recompute centroids
            old_centroids = self.centroids
            new_centroids = np.zeros((self.k, self.n_features))
            for c_i, c_idxs in enumerate(self.clusters):
                # assign new centroid value to the mean value of all samples that make up that cluster
                new_centroids[c_i] = np.mean(self.data[c_idxs], axis=0)
            self.centroids = new_centroids
            # print(old_centroids)
            # print(new_centroids)
            # break

            # check if algorithm converged
            centroid_dist = [calc_dist(self.centroids[i], old_centroids[i]) for i in range(self.k)]
            if np.sum(centroid_dist) == 0:
                print("Converged! " + str(it) + " iters")
                break
        # return cluster labels
        labels = np.zeros(self.n_samples).astype('int')
        for c_i, c_idxs in enumerate(self.clusters):
            for i in c_idxs:
                labels[i] = c_i + 1
        return labels