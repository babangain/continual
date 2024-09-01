import csv
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("google/code_x_glue_cc_code_to_code_trans")
test_set = dataset['test']

# Define file paths
generated_tsv_path = 'generated_cs_code.tsv'
generated_txt_path = 'cs_generated.txt'
reference_txt_path = 'cs_reference.txt'

# Read the generated C# code from the TSV file and write to cs_generated.txt
with open(generated_tsv_path, 'r', encoding='utf-8') as tsv_file:
    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    next(tsv_reader)  # Skip header row
    generated_cs_codes = [row[1] for row in tsv_reader]

with open(generated_txt_path, 'w', encoding='utf-8') as gen_file:
    for cs_code in generated_cs_codes:
        # Strip leading and trailing whitespace, replace newline characters with spaces
        cleaned_cs_code = cs_code.strip().replace('\n', ' ')
        gen_file.write(cleaned_cs_code + '\n')

# Write the reference C# code from the test set to cs_reference.txt
with open(reference_txt_path, 'w', encoding='utf-8') as ref_file:
    for example in test_set:
        # Strip leading and trailing whitespace, replace newline characters with spaces
        cleaned_cs_code = example['cs'].strip().replace('\n', ' ')
        ref_file.write(cleaned_cs_code + '\n')

print("Generated C# code has been written to cs_generated.txt")
print("Reference C# code has been written to cs_reference.txt")
