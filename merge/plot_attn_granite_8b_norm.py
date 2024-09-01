import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Load models and tokenizers
model_base = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-8b-code-base")
model_ft = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-8b-code-instruct")

# Get the state dictionaries of both models
print("Loading state dicts")
weights_base = model_base.state_dict()
print("Base state_dict loaded")
del model_base
weights_ft = model_ft.state_dict()
print("Ft state_dict loaded")
del model_ft

# Initialize lists to store histogram bins
hist_bins = np.linspace(-0.1, 0.1, 1000)  # You can adjust the range and bin size as needed
hist_values_mlp = np.zeros(len(hist_bins) - 1)
hist_values_self_attn = np.zeros(len(hist_bins) - 1)

# Initialize counters for the total number of weights
total_weights_mlp = 0
total_weights_self_attn = 0

# Compute the difference between the weights and update histogram values in chunks
for name, param in weights_base.items():
    print(name)
    if name in weights_ft:
        diff = (weights_ft[name] - param).cpu().numpy().flatten()
        hist, _ = np.histogram(diff, bins=hist_bins)
        if "mlp" in name:
            hist_values_mlp += hist
            total_weights_mlp += len(diff)
        elif "self_attn" in name:
            hist_values_self_attn += hist
            total_weights_self_attn += len(diff)
    else:
        print("The key is absent:", name)

# Normalize histograms to show percentage
hist_values_mlp = (hist_values_mlp / total_weights_mlp) * 100
hist_values_self_attn = (hist_values_self_attn / total_weights_self_attn) * 100

# Plot the histograms of delta values
fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

# MLP histogram
axs[0].bar(hist_bins[:-1], hist_values_mlp, width=np.diff(hist_bins), edgecolor='black', align='edge')
axs[0].set_title("Histogram of Delta Values (MLP)")
axs[0].set_xlabel("Delta Weight Value")
axs[0].set_ylabel("Percentage of Weights")

# Self Attention histogram
axs[1].bar(hist_bins[:-1], hist_values_self_attn, width=np.diff(hist_bins), edgecolor='black', align='edge')
axs[1].set_title("Histogram of Delta Values (Self Attention)")
axs[1].set_xlabel("Delta Weight Value")

plt.tight_layout()
plt.savefig("granite_8b_attn_ffn_side_by_side_percentage.png")
plt.show()
