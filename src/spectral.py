import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from kmeans import *

np.set_printoptions(suppress=True)

class Spectral:
    # constructor
    def __init__(self, data, q=25):
        self.data = data
        self.q = q
        self.n_samples, self.n_features = self.data.shape
        # Laplacian for the data
        self.L = self.init_laplacian()
    
    def init_laplacian(self):
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
        # print(S)
        # gets threshold of Qth percentile of distances in S
        q = np.percentile(S, self.q)
        # print("q:  " + str(q))
        # set a threshold distance, create adjacency matrix A (NxN)
        A = np.where(S < q, 1, 0)
        # print(A)
        ## trying to draw the graph
        # G = nx.from_numpy_matrix(np.array(A))  
        # pos = nx.spring_layout(G)
        # nx.draw_networkx_nodes(G, pos)
        # nx.draw_networkx_labels(G, pos)
        # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        # plt.show()

        np.fill_diagonal(A, 0)
        # find degree matrix by summing connections of each node in A
        D = np.diag(np.sum(A, axis=0))
        # print(D)
        # Create laplacian using L = D - A
        L = D - A
        # print(L)
        # print("created laplacian")
        return L

    
    def run(self):
        print("running spectral...")
        # find eigenvalues and eigenvectors of the Laplacian matrix
        eig_vals, eig_vecs = np.linalg.eig(self.L)

        # sort from lowest to highest eigenvalues
        eig_vals = eig_vals[np.argsort(eig_vals)]
        eig_vecs = eig_vecs[:, np.argsort(eig_vals)]
        # print("eigenvalues:")
        # np.set_printoptions(suppress=False)
        # print(eig_vals)
        # np.set_printoptions(suppress=True)
        # print("eigenvectors:")
        # print(eig_vecs)

        eig_difs = np.zeros_like(eig_vals)
        for i in range(1, len(eig_vals)):
            eig_difs[i] = eig_vals[i] - eig_vals[i-1]
        # print("eig difs")
        # print(eig_difs)
        # print(np.arange(len(eig_difs)))
        k = np.argmax(eig_difs)
        print(f'k: {k}')

        # Find index of first nonzero eigenvector
        i = np.where(eig_vals > 0.00001)[0][0]
        print(f'i: {i}')
        print(f'n: {len(eig_vals)}')
        k_eigs = eig_vecs[:, i:(k+i)]
        # print(k_eigs)

        kmeans = Kmeans(k_eigs, k)
        pred_labels = kmeans.run()
        return pred_labels

        # # Plot eigenvalues
        # plt.scatter(range(len(eig_vals)), eig_vals)
        # plt.show()