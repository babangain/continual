import os
import json
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def read_lines(file_path, num_lines=10000):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    selected_lines = lines[:num_lines]
    return selected_lines

# Root directory where the datasets are located
root_dir = '../../datasets'  # Change this to your actual datasets folder path

train_examples_list = []
test_examples_list = []

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
        file_path = os.path.join(subdir, file)
        if file == 'train.jsonl':
            random_lines = read_lines(file_path, num_lines=10000)
            train_examples_list.extend(random_lines)
        elif file == 'test.jsonl':
            random_lines = read_lines(file_path, num_lines=1000)
            test_examples_list.extend(random_lines)

# Ensure we have exactly 60,000 examples for training
train_examples_list = train_examples_list[:60000]

print(f"Total training examples collected: {len(train_examples_list)}")
print(f"Total test examples collected: {len(test_examples_list)}")

# Load the embeddings for training
embeddings = np.load('embeddings_large.npy')

# Perform k-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(embeddings)

# Get cluster labels for each training example
train_cluster_labels = kmeans.labels_

# Dictionary to store examples by cluster for training data
train_clustered_data = {0: [], 1: [], 2: []}

# Assign training examples to their respective clusters
for idx, label in enumerate(train_cluster_labels):
    train_clustered_data[label].append(train_examples_list[idx])

# Save clustered training data to separate JSONL files
train_output_dir = './train_clustered_data'  # Change to your desired output directory
os.makedirs(train_output_dir, exist_ok=True)

for cluster_id in range(3):
    output_file = os.path.join(train_output_dir, f'cluster_{cluster_id + 1}.jsonl')
    with open(output_file, 'w') as f:
        for line in train_clustered_data[cluster_id]:
            json_line = json.dumps({"text": line.strip()})
            f.write(json_line + '\n')

# Load the embeddings for test data
test_embeddings = np.load('embeddings_large_test.npy')

# Predict cluster labels for test data
test_cluster_labels = kmeans.predict(test_embeddings)

# Dictionary to store examples by cluster for test data
test_clustered_data = {0: [], 1: [], 2: []}

# Assign test examples to their respective clusters
for idx, label in enumerate(test_cluster_labels):
    test_clustered_data[label].append(test_examples_list[idx])

# Save clustered test data to separate JSONL files
test_output_dir = './test_clustered_data'  # Change to your desired output directory
os.makedirs(test_output_dir, exist_ok=True)

for cluster_id in range(3):
    output_file = os.path.join(test_output_dir, f'cluster_{cluster_id + 1}.jsonl')
    with open(output_file, 'w') as f:
        for line in test_clustered_data[cluster_id]:
            json_line = json.dumps({"text": line.strip()})
            f.write(json_line + '\n')

print("Data saved to JSONL files successfully.")
