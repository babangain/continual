import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import re
from tqdm import tqdm
import json
def extract_first_solution(solutions):
    if not solutions:
        return ""
    list_of_solutions = json.loads(solutions)
    return list_of_solutions[0]
# Load the dataset
dataset = load_dataset("codeparrot/apps", 'all')
train_set = dataset['train']
test_set = dataset['test']
train_set = train_set.map(lambda x: {'solutions': extract_first_solution(x['solutions'])})
test_set = test_set.map(lambda x: {'solutions': extract_first_solution(x['solutions'])})
print(test_set[0]['solutions'])

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizer
model_path = "ibm-granite/granite-3b-code-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()


# Randomly pick two examples from the training set for few-shot prompting
random.seed(42)
few_shot_examples = random.sample(list(train_set), 2)

# Create the input text from few-shot examples
few_shot_text = ""
for example in few_shot_examples:
    few_shot_text += f"Question: {example['question']}\nSolution: {example['solutions']}\n"

# Define batch size
batch_size = 1

# Function to generate translations for a batch
def generate_batch(batch,batch_ref,batch_starter_code):
    input_texts = [
        few_shot_text + f"Question: {example}\nSolution: "
        for example, starter_code in zip(batch, batch_starter_code)
    ]   
    input_tokens = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    output_tokens = model.generate(**input_tokens, max_new_tokens=512)
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    # Extract the Java code part from the outputs
    results = []
    for i, test_example in enumerate(batch):
        generated = output_texts[i].split("Solution:")[-1].strip().replace("\n", " ")
        results.append({"input": test_example,'reference': batch_ref[i], "generated": generated})
        
    return results

# Write results to a TSV file and text files
with open('codeparrot.tsv', 'w', newline='', encoding='utf-8') as tsv_file, \
     open('codeparrot_reference.txt', 'w', encoding='utf-8') as ref_file, \
     open('codeparrot_generated.txt', 'w', encoding='utf-8') as gen_file:
    
    tsv_writer = csv.writer(tsv_file, delimiter='\t')
    tsv_writer.writerow(["Question", "Reference", "Generated"])
    
    # Process the test set in batches and write results
    for i in tqdm(range(0, len(test_set), batch_size)):
        batch = test_set[i:i+batch_size]  # Batch contains full test examples now
        batch_results = generate_batch(batch['question'],batch['solutions'], batch['starter_code'])
        for result in batch_results:
            tsv_writer.writerow([result['input'], result['reference'], result['generated']])
            ref_file.write(result['reference'].strip().replace("\n", " ") + "\n")
            gen_file.write(result['generated'].strip().replace("\n", " ") + "\n")
            tsv_file.flush()
            ref_file.flush()
            gen_file.flush()

print("Results have been written.")
