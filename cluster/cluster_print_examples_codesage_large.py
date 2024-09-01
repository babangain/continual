import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def read_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    selected_lines = lines[:10000]
    return selected_lines

# Root directory where the datasets are located
root_dir = '../../datasets'  # Change this to your actual datasets folder path

examples_list = []

def index_to_lang(idx):
    if idx < 10000:
        return "go"
    if idx < 20000:
        return "java"
    if idx < 30000:
        return "javascript"
    if idx < 40000:
        return "php"
    if idx < 50000:
        return "python"
    if idx < 60000:
        return "ruby"
    return "error"

# Recursively iterate through all files in the datasets folder
for subdir, dirs, files in sorted(os.walk(root_dir)):
    for file in files:
        if file == 'train.jsonl':
            file_path = os.path.join(subdir, file)
            random_lines = read_lines(file_path)
            examples_list.extend(random_lines)

# Ensure we have exactly 60,000 examples
examples_list = examples_list[:60000]

print(f"Total examples collected: {len(examples_list)}")

# Load the embeddings
embeddings = np.load('embeddings_large.npy')

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(embeddings)

# Get cluster centroids
centroids = kmeans.cluster_centers_

# Print centroids
print("Cluster Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: {centroid}")

# Get cluster labels for each example
cluster_labels = kmeans.labels_

# Find the 5 nearest examples to each centroid
for cluster_id in range(3):
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_embeddings = embeddings[cluster_indices]
    distances = cdist(centroids[cluster_id].reshape(1, -1), cluster_embeddings).flatten()
    closest_examples_indices = np.argsort(distances)[:5]
    
    print(f"\nCluster {cluster_id+1} Examples:")
    for idx in closest_examples_indices:
        print(examples_list[cluster_indices[idx]])
