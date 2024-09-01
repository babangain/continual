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
    few_shot_text += f"C# code: {example['cs']}\nJava code: {example['java']}\n"

# Define batch size
batch_size = 16

# Function to generate translations for a batch
def generate_batch(batch,batch_ref):
    input_texts = [few_shot_text + f"C# code: {example}\nJava code: " for example in batch]
    input_tokens = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    output_tokens = model.generate(**input_tokens, max_new_tokens=512)
    output_texts = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    
    # Extract the Java code part from the outputs
    results = []
    for i, test_example in enumerate(batch):
        generated_java_code = output_texts[i].split("Java code:")[-1].strip().replace("\n", " ")
        results.append({"cs": test_example,'java_ref': batch_ref[i], "generated_java": generated_java_code})
        
    return results

# Write results to a TSV file and text files
with open('generated_java_code.tsv', 'w', newline='', encoding='utf-8') as java_file, \
     open('java_reference.txt', 'w', encoding='utf-8') as java_file_ref, \
     open('java_generated.txt', 'w', encoding='utf-8') as generated_java_file:
    
    tsv_writer = csv.writer(java_file, delimiter='\t')
    tsv_writer.writerow(["C# Code", "Java Reference Code", "Generated Java Code"])
    
    # Process the test set in batches and write results
    for i in tqdm(range(0, len(test_set), batch_size)):
        batch = test_set[i:i+batch_size]  # Batch contains full test examples now
        batch_results = generate_batch(batch['cs'],batch['java'])
        for result in batch_results:
            tsv_writer.writerow([result['cs'], result['java_ref'], result['generated_java']])
            java_file_ref.write(result['java_ref'].strip().replace("\n", " ") + "\n")
            generated_java_file.write(result['generated_java'].strip().replace("\n", " ") + "\n")
            java_file.flush()
            java_file_ref.flush()
            generated_java_file.flush()

print("Results have been written to generated_java_code.tsv, java_reference.txt, and java_generated.txt")
