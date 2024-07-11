import sys
import os

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest

from dexire_pro.core.clustering_explanations import ClusterAnalysis

@pytest.fixture
def generate_data():
    # Generate synthetic data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return X


def test_clustering(generate_data):
# Initialize list to store within-cluster sum of squares
    X = generate_data
    ca = ClusterAnalysis()
    ca.automatically_choose_cluster_numbers(X, max_clusters=15)
    assert ca.cluster_model is not None