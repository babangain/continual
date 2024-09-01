import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Load models and tokenizers
tokenizer_base = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model_base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenizer_ft = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Python-hf")
model_ft = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-Python-hf")

# Get the state dictionaries of both models
print("Loading state dicts")
weights_base = model_base.state_dict()
print("Base state_dict loaded")
del model_base
weights_ft = model_ft.state_dict()
print("Ft state_dict loaded")
del model_ft

# Initialize a list to store histogram bins
hist_bins = np.linspace(-100, 100, 10000)  # Percentage change range from -100% to +100%
hist_values = np.zeros(len(hist_bins) - 1)

# Compute the percentage difference between the weights and update histogram values in chunks
for name, param in weights_base.items():
    print(name)
    if name in weights_ft:
        base_param = param.cpu().numpy().flatten()
        ft_param = weights_ft[name].cpu().numpy().flatten()
        
        # Avoid division by zero by adding a small epsilon value
        epsilon = 1e-8
        base_param = np.where(base_param == 0, epsilon, base_param)
        
        # Calculate percentage change
        diff = ((ft_param - base_param) / base_param) * 100
        hist, _ = np.histogram(diff, bins=hist_bins)
        hist_values += hist
    else:
        print("The key is absent:", name)

# Plot the histogram of percentage delta values
plt.figure(figsize=(10, 6))
plt.bar(hist_bins[:-1], hist_values, width=np.diff(hist_bins), edgecolor='black', align='edge')
plt.title("Histogram of Percentage Delta Values")
plt.xlabel("Percentage Delta Weight Value (%)")
plt.ylabel("Frequency")
plt.savefig("llama_7b_percentage_diff.png")
plt.show()
