# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array


class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        X = check_array(X, accept_sparse='csr')

        np.random.seed(self.random_state)
        self.cluster_centers_ = self.initializeCentroids(X)

        for iteration in range(self.max_iter):
            distances = self.calculateDistances(X)
            self.labels_ = self.assignPoints(distances)

            new_centers = self.updateCentroids(X)

            if self.checkConvergence(new_centers):
                break

            self.cluster_centers_ = new_centers

        return self

    def initializeCentroids(self, data):
        return data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]

    def calculateDistances(self, data):
        distances = np.linalg.norm(data[:, None] - self.cluster_centers_, axis=2)
        return distances

    def assignPoints(self, distances):
        return np.argmin(distances, axis=1)

    def updateCentroids(self, data):
        new_centers = np.array([data[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers

    def checkConvergence(self, new_centers):
        return np.allclose(new_centers, self.cluster_centers_, atol=1e-4)


    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_


class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances (optional).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def fit(self, data):
        assert isinstance(data, np.ndarray), DataFormatError("Invalid input")
        assert len(data.shape) == 2, ShapeError("Invalid input")

        assigned_clusters = np.full(len(data), -1)
        cluster_count = 0
        eps = self.eps  
        min_samples = self.min_samples
        for i, point in enumerate(data):
            if assigned_clusters[i] != -1:
                continue

            num_nearby_points = np.sum(np.linalg.norm(data - point, axis=1) < eps)

            if num_nearby_points < min_samples:
                assigned_clusters[i] = -1
                continue

            assigned_clusters[i] = cluster_count
            stack = [i]

            while stack:
                current_idx = stack.pop()
                current_nearby = np.sum(np.linalg.norm(data - data[current_idx], axis=1) < eps)

                if current_nearby >= min_samples:
                    neighborhood_indices = np.where(current_nearby)[0]
                    for idx in neighborhood_indices:
                        if assigned_clusters[idx] == -1:
                            assigned_clusters[idx] = cluster_count
                            stack.append(idx)
                else:
                    continue

            cluster_count += 1

        self.labels_ = assigned_clusters
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
