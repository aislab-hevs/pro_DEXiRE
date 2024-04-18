import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import Tuple
from yellowbrick.cluster import KElbowVisualizer
from pathlib import Path
import pickle
import umap
from enum import Enum

class DimensionalityReduction(str, Enum):
    PCA = "PCA"
    TSNE = "TSNE"
    UMAP = "UMAP"
    KernelPCA = "KernelPCA"
    
    

class ClusterAnalysis:
    def __init__(self, 
                 n_cluster:int=None, 
                 normalize: bool = True) -> None:
        self.n_clusters = n_cluster
        self.cluster_model = None
        self.wcss = []
        self.normalize = normalize
        self.scaler = StandardScaler()
    
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
    
    def automatically_choose_cluster_numbers(self, 
                                             X: np.array, 
                                             max_clusters: int = 11) -> bool:
        if self.normalize:
            X = self.scaler.fit_transform(X)
        possible_n_clusters = np.arange(2, max_clusters)
        for i in possible_n_clusters:
            kmeans = KMeans(n_clusters=i, init='k-means++', 
                            max_iter=300, 
                            n_init=10, 
                            random_state=0)
            kmeans.fit(X)
            self.wcss.append(kmeans.inertia_)
        self.plot_elbow(X, max_cluster=max_clusters)
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
        
    def plot_elbow(self, 
                   X: np.array, 
                   out_file: Path = None, 
                   max_cluster: int = 11) -> None:
        kmeans = KMeans()
        visualizer = KElbowVisualizer(kmeans, k=(2, max_cluster))
        visualizer.fit(X)  # Fit the data to the visualizer
        if out_file is not None:
            visualizer.show(outpath=out_file)
        else:
            visualizer.show()
    
    def plot_clusters(self, 
                      X: np.array, 
                      y: np.array, 
                      dimensionality_reduction: str = DimensionalityReduction.PCA):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y have different shapes")
        if dimensionality_reduction == DimensionalityReduction.PCA:
            pca = PCA(n_components=2)
            X_new = pca.fit_transform(X)
        elif dimensionality_reduction == DimensionalityReduction.TSNE:
            tsne = TSNE(n_components=2, random_state=0)
            X_new = tsne.fit_transform(X)
        elif dimensionality_reduction == DimensionalityReduction.UMAP:
            reducer = umap.UMAP(n_components=2, random_state=0)
            X_new = reducer.fit_transform(X)
        elif dimensionality_reduction == DimensionalityReduction.KernelPCA:
            kpca = KernelPCA(n_components=2, kernel='rbf')
            X_new = kpca.fit_transform(X)
        else:
            raise ValueError(f"Dimensionality reduction {dimensionality_reduction} is not supported.")
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
        plt.show()
    
    def save_cluster_model(self, file_path:str):
        #TODO: Implement save_cluster_model method
        with open(file_path, 'w') as f:
            pickle.dump(self.cluster_model, f)
    
    def load_cluster_model(self,):
        #TODO: Implement load_cluster_model method
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