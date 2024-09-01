import os
import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import sys

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Initialize the tokenizer and model
checkpoint = "codesage/codesage-base"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True,add_eos_token=True)
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True).to(device)

# List to store embeddings
embeddings_list = []

# Define the root directory
root_dir = '../../datasets'

# Function to extract random lines from a file
def read_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    selected_lines = lines[:10000]
    return selected_lines

# Recursively iterate through all files in the datasets folder
for subdir, dirs, files in sorted(os.walk(root_dir)):
    for file in files:
        if file == 'train.jsonl':
            file_path = os.path.join(subdir, file)
            random_lines = read_lines(file_path)
            for i, line in enumerate(tqdm(random_lines)):
                try:
                    data = json.loads(line)
                    if 'input' in data:
                        input_text = data['input']
                        cleaned_text = input_text.replace("Summarize the following code: ", "")

                        # Tokenize and get embeddings
                        inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=2048).to(device)

                        # Check tensor shapes and values
                        
                        #print("Line no: ", i )
                        #input_length = inputs['input_ids'].shape[1]
                        #print(f"Length of the input: {input_length}")

                        with torch.no_grad():
                            outputs = model(**inputs)
                            embedding = outputs.pooler_output
                        
                        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                            print("Tensor contains NaNs or Infs")
                            print("Line no: ", i )
                            continue
                        
                        embeddings_list.append(embedding.detach().cpu().numpy())
                except Exception as e:
                    print(f"Error at line {i}: {e}")
                    sys.exit(0)

# Convert embeddings list to numpy array for clustering
embeddings_array = np.vstack(embeddings_list)

# Save the embeddings array to a file
np.save('embeddings.npy', embeddings_array)
