import os
import json
import torch
import numpy as np
import random
from transformers import AutoModel, AutoTokenizer
from sklearn.cluster import KMeans
from tqdm import tqdm
# Initialize the tokenizer and model
checkpoint = "codesage/codesage-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to('cuda' if torch.cuda.is_available() else 'cpu')

# List to store embeddings
embeddings_list = []

# Define the root directory
root_dir = '../datasets'

# Function to extract random lines from a file
def extract_random_lines(file_path, num_lines=1000):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    random.shuffle(lines)
    selected_lines = lines[:num_lines]
    return selected_lines

# Recursively iterate through all files in the datasets folder
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'train.jsonl':
            file_path = os.path.join(subdir, file)
            random_lines = extract_random_lines(file_path)
            for line in tqdm(random_lines):
                data = json.loads(line)
                if 'input' in data:
                    input_text = data['input']
                    cleaned_text = input_text.replace("Summarize the following code: ", "")
                    #print(input_text)
                    
                    # Tokenize and get embeddings
                    inputs = tokenizer(cleaned_text, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')
                    with torch.no_grad():
                        embedding = model(**inputs).pooler_output
                    
                    # Store the embedding
                    embeddings_list.append(embedding.cpu().numpy())

# Convert embeddings list to numpy array for clustering
embeddings_array = np.vstack(embeddings_list)

# Save the embeddings array to a file
np.save('embeddings.npy', embeddings_array)

# Load the embeddings array from the file
loaded_embeddings = np.load('embeddings.npy')

# Cluster the embeddings using KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(loaded_embeddings)

# Print the cluster assignments
print("Cluster assignments:", kmeans.labels_)

# Optionally, you can print the cluster centers
print("Cluster centers:", kmeans.cluster_centers_)
