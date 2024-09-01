import os
from sklearn.metrics import pairwise_distances_argmin_min,pairwise_distances
from scipy.spatial.distance import pdist
from yellowbrick.cluster import KElbowVisualizer

examples_list = []
def read_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    selected_lines = lines[:10000]
    return selected_lines

# Root directory where the datasets are located
root_dir = '../../datasets'  # Change this to your actual datasets folder path

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


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Load the embeddings
embeddings = np.load('embeddings.npy')

# Perform k-means clustering with 3 clusters
kmeans = KMeans(random_state=0)
visualizer = KElbowVisualizer(kmeans, k=(2,10),metric="silhouette")
visualizer.fit(embeddings)
visualizer.show(outpath='elbow_plot_silhouette.png')
