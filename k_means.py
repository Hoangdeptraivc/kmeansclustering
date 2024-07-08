import numpy as np
import matplotlib.pyplot as plt

np.random.seed(10)
means = [[2, 3], [6, 3], [3, 7]]
cov = [[1, 0], [0, 1]]
n = 500
x0 = np.random.multivariate_normal(means[0], cov, n)
x1 = np.random.multivariate_normal(means[1], cov, n)
x2 = np.random.multivariate_normal(means[2], cov, n)
x = np.concatenate((x0, x1, x2))

class KMeans:
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.centroids = None
        self.labels = None

    def initialize_centroids(self):
        indices = np.random.choice(len(self.x), self.k, replace=False)
        self.centroids = self.x[indices]

    def assign_clusters(self):
        distances = np.sqrt(np.sum((self.x[:, np.newaxis] - self.centroids)**2, axis=-1))
        self.labels = np.argmin(distances, axis=1)

    def update_centroids(self):
        new_centroids = np.array([self.x[self.labels == i].mean(axis=0) for i in range(self.k)])
        self.centroids = new_centroids

    def fit(self, max_iter=100):
        self.initialize_centroids()
        for _ in range(max_iter):
            old_centroids = self.centroids.copy()
            self.assign_clusters()
            self.update_centroids()
            if np.allclose(old_centroids, self.centroids):
                break

    def plot_clusters(self):
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i in range(self.k):
            cluster_points = self.x[self.labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i+1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', color='black', label='Centroids')
        plt.legend()
        plt.title('K-means Clustering')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

def main():
    k = 3
    kmeans = KMeans(x, k)
    kmeans.fit()
    kmeans.plot_clusters()

if __name__ == "__main__":
    main()
    
