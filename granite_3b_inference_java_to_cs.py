import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans")
train_set = dataset['train']
test_set = dataset['test']

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model_path = "ibm-granite/granite-3b-code-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

# Randomly pick two examples from the training set for few-shot prompting
few_shot_examples = random.sample(list(train_set), 2)

# Create the input text from few-shot examples
few_shot_text = ""
for example in few_shot_examples:
    few_shot_text += f"Java code: {example['java']}\nC# code: {example['cs']}\n"
#few_shot_text = few_shot_text + "Based on the aforementioned examples, translate the following code.\n"
# Define batch size
batch_size = 8

# Function to generate translations for a batch
def generate_batch(batch):
    input_texts = [few_shot_text + f"Java code: {example}\nC# code: " for example in batch]
    input_tokens = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    output_tokens = model.generate(**input_tokens, max_new_tokens=512)
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    # Extract the C# code part from the outputs
    results = []
    for i, test_example in enumerate(batch):
        generated_cs_code = output_texts[i].split("C# code:")[-1].strip()
        results.append({"java": input_texts[i], "generated_cs": generated_cs_code})
        #print("Input_text: ", input_texts[i])
        #print("Output: ", output_texts)

    return results

# Convert the test set to a list of Java code strings
test_java_codes = test_set['java']

# Write results to a TSV file
with open('generated_cs_code.tsv', 'w', newline='', encoding='utf-8') as file:
    tsv_writer = csv.writer(file, delimiter='\t')
    tsv_writer.writerow(["Java Code", "Generated C# Code"])
    
    # Process the test set in batches and write results
    for i in tqdm(range(0, len(test_java_codes), batch_size)):
        batch = test_java_codes[i:i+batch_size]
        batch_results = generate_batch(batch)
        for result in batch_results:
            tsv_writer.writerow([result['java'], result['generated_cs']])
            file.flush()

print("Results have been written to generated_cs_code.tsv")
