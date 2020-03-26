import math
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering


class NcutSegmenter:

    def __init__(self, values, k):
        self.values = values
        self.k = k

    def customNcuts(self):
        """ Return segmentation label using classic Ncuts """
        # computing neighboors graph
        A = kneighbors_graph(self.values, self.k, mode='distance', include_self=False).toarray()

        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[0]):
                if A[i][j] > 0:

                    v1 = (self.values[i][3], self.values[i][4], self.values[i][5])
                    v2 = (self.values[j][3], self.values[j][4], self.values[j][5])

                    magnitude1 = np.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
                    magnitude2 = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
                    ang = np.arccos(np.dot(v1, v2) / (magnitude1 * magnitude2))

                    A[i][j] = max(self.values[i][7], self.values[j][7]) * A[i][j]

        # init SpectralClustering
        sc = SpectralClustering(4, affinity='precomputed', n_init=10, assign_labels = 'discretize')

        # cluster
        labels = sc.fit_predict(A)

        return labels

    def segment_func1(self):
        """ Return SpectralClustering segmentation label using the normal angle edge weights"""
        # computing neighboors graph
        A = self.normal_graph()

        # SpectralClustering segmentation
        sc = SpectralClustering(3, affinity='precomputed', n_init=10, assign_labels='discretize')
        labels = sc.fit_predict(A)

        return labels

    def normal_graph(self):
        """ Calculates the graph matrix using the angle between normals as node weight """
        A = kneighbors_graph(self.values, self.k, mode='connectivity', include_self=False).toarray()

        print("  creating affinity matrix A (Normal angles)")

        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[0]):

                if A[i][j] == 1:

                    v1 = (self.values[i][3], self.values[i][4], self.values[i][5])
                    v2 = (self.values[j][3], self.values[j][4], self.values[j][5])

                    magnitude1 = np.sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])
                    magnitude2 = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
                    ang = np.arccos(np.dot(v1, v2) / (magnitude1 * magnitude2))

                    A[i][j] = ang
        print("  Done.")
        return A

    def segment_func2(self):
        """ Return SpectralClustering segmentation label using the bondary prediction as edge weights"""
        # computing neighboors graph
        A = self.boundaryprob_graph()

        # SpectralClustering segmentation
        sc = SpectralClustering(3, affinity='precomputed', n_init=10, assign_labels='discretize')
        labels = sc.fit_predict(A)

        return labels

    def boundaryprob_graph(self):
        A = kneighbors_graph(self.values, self.k, mode='connectivity', include_self=False).toarray()

        print("  creating affinity matrix A (boundary prob)")

        for i in range(self.values.shape[0]):
            for j in range(self.values.shape[0]):
                if A[i][j] == 1:
                    A[i][j] = max(self.values[i][7], self.values[j][7])
        print("  Done.")
        return A

