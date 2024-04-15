import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from typing import Tuple
from yellowbrick.cluster import KElbowVisualizer
from pathlib import Path
import pickle
from enum import Enum

class DimensionalityReduction(str, Enum):
    PCA = "PCA"
    TSNE = "TSNE"
    UMAP = "UMAP"
    KernelPCA = "KernelPCA"
    
    

class ClusterAnalysis:
    def __init__(self, n_cluster:int=None) -> None:
        self.n_clusters = n_cluster
        self.cluster_model = None
        self.wcss = []
    
    def fit(self, X: np.array, n_clusters: int = None) -> None:
        if n_clusters is not None and n_clusters > 1:
            self.n_clusters = n_clusters
            self.cluster_model = KMeans(n_clusters=n_clusters)
            self.cluster_model.fit(X)
    
    def predict(self, X: np.array) -> np.array:
        """This function predict 

        :param X: _description_
        :type X: np.array
        :raises Exception: _description_
        :return: _description_
        :rtype: np.array
        """
        if self.cluster_model is not None:
            try:
                check_is_fitted(self.cluster_model)
            except NotFittedError as exc:
                print(f"The  model have not been fitted and cannot be used for prediction.")
                raise Exception("The cluster number have not been set. Model cannot be fitted.")
            return self.cluster_model.predict(X)
        
    
    def score(self, X: np.array, y: np.array) -> float:
        #TODO: Implement score method
        pass
    
    def automatically_choose_cluster_numbers(self, 
                                             X: np.array, 
                                             max_clusters: int = 11) -> bool:
        
        possible_n_clusters = np.arange(2, max_clusters)
        for i in possible_n_clusters:
            kmeans = KMeans(n_clusters=i, init='k-means++', 
                            max_iter=300, 
                            n_init=10, 
                            random_state=0)
            kmeans.fit(X)
            self.wcss.append(kmeans.inertia_)
        # Calculate the deltas with first derivate
        delta_wcss = np.diff(self.wcss)
        # Second derivate difference 
        delta2_wcss = np.diff(delta_wcss)
        # Find the index of the elbow point
        elbow_index = np.where(delta2_wcss > 0)[0][0] + 2
        best_cluster = possible_n_clusters[elbow_index]
        self.n_clusters = best_cluster
        # fit the model 
        self.fit(X, self.n_clusters)
        return True
        
    def plot_elbow(self, X: np.array, out_file: Path = None, max_cluster: int = 11) -> None:
        kmeans = KMeans()
        visualizer = KElbowVisualizer(kmeans, k=(2, max_cluster))
        visualizer.fit(X)  # Fit the data to the visualizer
        if out_file is not None:
            visualizer.show(outpath=out_file)
        else:
            visualizer.show()
    
    def plot_clusters(self, X, y, dimensionality_reduction: str = DimensionalityReduction.PCA):
        #TODO: Implement plot_clusters method
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different shapes")
        if dimensionality_reduction == DimensionalityReduction.PCA:
            pca = PCA(n_components=2)
            X_new = pca.fit_transform(X)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        scatter = ax.scatter(
            X_new[:, 0],
            X_new[:, 1],
            c=y,
            cmap='viridis',
            edgecolor='k',
            s=50
        )
        ax.set_title('Clusters')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        plt.colorbar(scatter)
    
    def save_cluster_model(self,):
        #TODO: Implement save_cluster_model method
        pass
    



# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Initialize list to store within-cluster sum of squares
wcss = []

# Try different numbers of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    # Append the WCSS for the current number of clusters to the list
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()