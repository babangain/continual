import os
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from scipy.stats import kstest

# Create directory for saving figures
output_dir = "figs-llama-mean-std"
os.makedirs(output_dir, exist_ok=True)

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

# Initialize histogram bins
hist_bins = np.linspace(-0.1, 0.1, 1000)  # You can adjust the range and bin size as needed

# Function to plot and save histogram
def plot_and_save_histogram(diff, layer_name):
    hist, _ = np.histogram(diff, bins=hist_bins)
    hist_percentage = (hist / len(diff)) * 100  # Normalize to percentage

    # Calculate mean and standard deviation
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    # Perform KS test
    ks_stat, p_value = kstest(diff, 'norm', args=(mean_diff, std_diff))

    plt.figure(figsize=(10, 6))
    plt.bar(hist_bins[:-1], hist_percentage, width=np.diff(hist_bins), edgecolor='black', align='edge')
    plt.title(f"Histogram of Delta Values ({layer_name})\nMean: {mean_diff:.6f}, Std Dev: {std_diff:.6f}, KS p-value: {p_value:.6f}")
    plt.xlabel("Delta Weight Value")
    plt.ylabel("Percentage of Weights")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"histogram_delta_values_{layer_name}.png"))
    plt.close()

# Iterate over each layer and compute the difference
for name, param in weights_base.items():
    if name in weights_ft:
        diff = (weights_ft[name] - param).cpu().numpy().flatten()
        if "mlp" in name or "self_attn" in name:
            layer_name = name.replace(".", "_")
            plot_and_save_histogram(diff, layer_name)
    else:
        print("The key is absent:", name)
