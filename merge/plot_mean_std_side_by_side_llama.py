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

# Function to plot and save histograms
def plot_and_save_histograms(base_weights, ft_weights, layer_name):
    base_hist, _ = np.histogram(base_weights, bins=hist_bins)
    ft_hist, _ = np.histogram(ft_weights, bins=hist_bins)

    base_hist_percentage = (base_hist / len(base_weights)) * 100  # Normalize to percentage
    ft_hist_percentage = (ft_hist / len(ft_weights)) * 100  # Normalize to percentage

    # Calculate mean and standard deviation
    mean_base = np.mean(base_weights)
    std_base = np.std(base_weights)
    mean_ft = np.mean(ft_weights)
    std_ft = np.std(ft_weights)

    # Perform KS test
    ks_stat_base, p_value_base = kstest(base_weights, 'norm', args=(mean_base, std_base))
    ks_stat_ft, p_value_ft = kstest(ft_weights, 'norm', args=(mean_ft, std_ft))

    plt.figure(figsize=(18, 6))
    
    # Plot for base weights
    plt.subplot(1, 2, 1)
    plt.bar(hist_bins[:-1], base_hist_percentage, width=np.diff(hist_bins), edgecolor='black', align='edge')
    plt.title(f"Histogram of Base Values ({layer_name})\nMean: {mean_base:.6f}, Std Dev: {std_base:.6f}, KS p-value: {p_value_base:.6f}")
    plt.xlabel("Base Weight Value")
    plt.ylabel("Percentage of Weights")
    
    # Plot for fine-tuned weights
    plt.subplot(1, 2, 2)
    plt.bar(hist_bins[:-1], ft_hist_percentage, width=np.diff(hist_bins), edgecolor='black', align='edge')
    plt.title(f"Histogram of Fine-tuned Values ({layer_name})\nMean: {mean_ft:.6f}, Std Dev: {std_ft:.6f}, KS p-value: {p_value_ft:.6f}")
    plt.xlabel("Fine-tuned Weight Value")
    plt.ylabel("Percentage of Weights")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"histogram_base_vs_ft_{layer_name}.png"))
    plt.close()

# Iterate over each layer and compute the histograms
for name, param in weights_base.items():
    if name in weights_ft:
        base_weights = param.cpu().numpy().flatten()
        ft_weights = weights_ft[name].cpu().numpy().flatten()
        if "mlp" in name or "self_attn" in name:
            layer_name = name.replace(".", "_")
            plot_and_save_histograms(base_weights, ft_weights, layer_name)
    else:
        print("The key is absent:", name)
