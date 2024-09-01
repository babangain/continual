import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import re
from tqdm import tqdm
def remove_triple_quotes(text):
    return re.sub(r'""".*?"""', '', text, flags=re.DOTALL)


# Load the dataset
dataset = load_dataset("google/code_x_glue_ct_code_to_text", 'python')
train_set = dataset['train']
test_set = dataset['test']

# Apply the function to the 'code' column of the test_set
test_set = test_set.map(lambda x: {'code': remove_triple_quotes(x['code'])})
train_set = train_set.map(lambda x: {'code': remove_triple_quotes(x['code'])})

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
    few_shot_text += f"Code: {example['code']}\nDocstring: {example['docstring']}\n"

# Define batch size
batch_size = 16

# Function to generate translations for a batch
def generate_batch(batch,batch_ref):
    input_texts = [few_shot_text + f"Code: {example}\nDocstring: " for example in batch]
    input_tokens = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    output_tokens = model.generate(**input_tokens, max_new_tokens=512)
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    # Extract the Java code part from the outputs
    results = []
    for i, test_example in enumerate(batch):
        generated = output_texts[i].split("Docstring:")[-1].strip().replace("\n", " ")
        results.append({"input": test_example,'reference': batch_ref[i], "generated": generated})
        
    return results

# Write results to a TSV file and text files
with open('python_docstring.tsv', 'w', newline='', encoding='utf-8') as tsv_file, \
     open('python_docstring_reference.txt', 'w', encoding='utf-8') as ref_file, \
     open('python_docstring_generated.txt', 'w', encoding='utf-8') as gen_file:
    
    tsv_writer = csv.writer(tsv_file, delimiter='\t')
    tsv_writer.writerow(["Code", "Reference Docstring", "Generated Docstring"])
    
    # Process the test set in batches and write results
    for i in tqdm(range(0, len(test_set), batch_size)):
        batch = test_set[i:i+batch_size]  # Batch contains full test examples now
        batch_results = generate_batch(batch['code'],batch['docstring'])
        for result in batch_results:
            tsv_writer.writerow([result['input'], result['reference'], result['generated']])
            ref_file.write(result['reference'].strip().replace("\n", " ") + "\n")
            gen_file.write(result['generated'].strip().replace("\n", " ") + "\n")
            tsv_file.flush()
            ref_file.flush()
            gen_file.flush()

print("Results have been written.")
