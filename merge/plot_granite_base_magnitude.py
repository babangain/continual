import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
import numpy as np

# Load models
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

# Define bins for the histogram
bins = np.linspace(-1, 1, 1000)
hist_values_base = np.zeros(len(bins) - 1)
hist_values_ft = np.zeros(len(bins) - 1)

# Compute histograms for weight values in chunks
for name, param in weights_base.items():
    print(name)
    if name in weights_ft:
        # Get the raw values of weights
        values_base = param.cpu().numpy().flatten()
        values_ft = weights_ft[name].cpu().numpy().flatten()

        # Update histograms
        hist_base, _ = np.histogram(values_base, bins=bins)
        hist_ft, _ = np.histogram(values_ft, bins=bins)
        
        hist_values_base += hist_base
        hist_values_ft += hist_ft
    else:
        print("The key is absent:", name)

# Plot the histograms of weight values side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.bar(bins[:-1], hist_values_base, width=np.diff(bins), color='blue', edgecolor='black', align='edge')
ax1.set_title("Base Model Weight Values")
ax1.set_xlabel("Weight Value")
ax1.set_ylabel("Frequency")
ax1.grid(True, which="both", ls="--")

ax2.bar(bins[:-1], hist_values_ft, width=np.diff(bins), color='red', edgecolor='black', align='edge')
ax2.set_title("Fine-Tuned Model Weight Values")
ax2.set_xlabel("Weight Value")
ax2.set_ylabel("Frequency")
ax2.grid(True, which="both", ls="--")

plt.tight_layout()
plt.savefig("fig_granite_8b_value_comparison_side_by_side.png")
plt.show()
