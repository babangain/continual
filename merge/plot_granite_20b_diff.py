import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Load models and tokenizers
model_base = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-20b-code-base")

model_ft = AutoModelForCausalLM.from_pretrained("ibm-granite/granite-20b-code-instruct")

# Get the state dictionaries of both models
print("Loading state dicts")
weights_base = model_base.state_dict()
print("Base  state_dict loaded")
del model_base
weights_ft = model_ft.state_dict()
print("Ft state_dict loaded")
del model_ft

# Initialize a list to store histogram bins
hist_bins = np.linspace(-0.1,0.1, 1000)  # You can adjust the range and bin size as needed
hist_values = np.zeros(len(hist_bins) - 1)

# Compute the difference between the weights and update histogram values in chunks
for name, param in weights_base.items():
    print(name)
    if name in weights_ft:
        diff = (weights_ft[name] - param).cpu().numpy().flatten()
        hist, _ = np.histogram(diff, bins=hist_bins)
        hist_values += hist
    else:
        print("The key is absent:", name)

# Plot the histogram of delta values
plt.figure(figsize=(10, 6))
plt.bar(hist_bins[:-1], hist_values, width=np.diff(hist_bins), edgecolor='black', align='edge')
plt.title("Histogram of Delta Values")
plt.xlabel("Delta Weight Value")
plt.ylabel("Frequency")
plt.savefig("granite_20b_diff.png")
plt.show()

