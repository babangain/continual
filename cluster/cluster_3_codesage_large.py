import os
from sklearn.metrics import pairwise_distances_argmin_min,pairwise_distances
from scipy.spatial.distance import pdist


def read_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    selected_lines = lines[:10000]
    return selected_lines

# Root directory where the datasets are located
root_dir = '../../datasets'  # Change this to your actual datasets folder path

examples_list = []
def index_to_lang(idx):
    if idx <10000:
        return "go"
    if idx < 20000:
        return "java"
    if idx < 30000:
        return "javascript"
    if idx < 40000:
        return "php"
    if idx <50000:
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


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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
# cluster_labels = kmeans.labels_

# for cluster_id in range(3):
#   # Get indices of all data points belonging to the current cluster
#   cluster_indices = np.where(cluster_labels == cluster_id)[0]

#   # Shuffle the indices to pick random examples
#   np.random.shuffle(cluster_indices)

#   # Print 5 random examples for the current cluster
#   print(f"\nCluster {cluster_id+1} Examples:")
#   for i in range(5):
#     example_index = cluster_indices[i]
#     print(examples_list[example_index])

# Get cluster labels for each example
cluster_labels = kmeans.labels_

# Dictionary to store counts of each language in each cluster
language_cluster_counts = {
    "go": [0, 0, 0],
    "java": [0, 0, 0],
    "javascript": [0, 0, 0],
    "php": [0, 0, 0],
    "python": [0, 0, 0],
    "ruby": [0, 0, 0]
}

# Determine cluster labels for each example and count occurrences
for idx, line in enumerate(examples_list):
    lang = index_to_lang(idx)
    cluster_id = cluster_labels[idx]
    language_cluster_counts[lang][cluster_id] += 1


# Plotting the bar chart
import matplotlib.pyplot as plt

clusters = ['Cluster 1', 'Cluster 2', 'Cluster 3']
bar_width = 0.15
index = np.arange(len(clusters))

fig, ax = plt.subplots(figsize=(12, 8))

for i, (lang, counts) in enumerate(language_cluster_counts.items()):
    ax.bar(index + i * bar_width, counts, bar_width, label=lang)

ax.set_xlabel('Clusters')
ax.set_ylabel('Count')
ax.set_title('Cluster Distribution of Programming Languages')
ax.set_xticks(index + bar_width * 2)
ax.set_xticklabels(clusters)
ax.legend()

plt.tight_layout()
plt.savefig('cluster_3_codesage_large.jpg')
plt.show()