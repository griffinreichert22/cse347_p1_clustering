import numpy as np
# import networkx as nx

def calc_dist(a, b):
    return np.linalg.norm(a-b)

class Spectral:
    # constructor
    def __init__(self, data, k=5):
        self.data = data
        self.k = k
        self.n_samples, self.n_features = self.data.shape
        # Laplacian for the data
        self.L = self.init_laplacian()
    
    def init_laplacian(self):
        print("init laplacian")
        # create similarity matrix (NxN) - distances from one point to another
        S = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(self.n_samples):
                # when i == j distance is 0
                dist = 0
                if i != j:
                    # calculate distance between point i and j
                    dist = calc_dist(self.data[i], self.data[j])
                S[i][j] = dist
                # distance is symmetric , use this to optimize calculation
                # distance between A and B is same as distance between B and A
                S[j][i] = dist
        print(S)
        # gets threshold of 25th percentile of distances in S
        q25 = np.percentile(S, 25)
        print("q25:  " + str(q25))
        # set a threshold distance, create adjacency matrix A (NxN)
        A = np.where(S < q25, 1, 0)
        print(A)
        # find degree matrix by summing connections of each node in A
        D = np.diag(np.sum(A, axis=0))
        print(D)
        # Create laplacian using L = D - A
        L = D - A
        print(L)
        return L

    
    def run(self):
        print("running spectral")