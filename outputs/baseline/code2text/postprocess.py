import pandas as pd
import re

def process_docstring(docstring):
    # Check if the docstring starts with "Code:"
    docstring = str(docstring)
    if docstring.startswith("Code:"):
        # Try to find triple quotes
        match = re.search(r'("""|\'\'\')', docstring)
        if match:
            docstring = docstring[match.start():]
        else:
            return ""
    
    # Remove any "Code:" after encountering the docstring
    code_index = docstring.find("Code:")
    if code_index != -1:
        docstring = docstring[:code_index].strip()
    
    # Replace multiple spaces with a single space
    docstring = re.sub(r'\s+', ' ', docstring)
    
    return docstring.strip()

def process_tsv(input_file, output_file):
    # Read the TSV file
    df = pd.read_csv(input_file, sep='\t')
    
    # Process each generated docstring
    df['Generated Docstring'] = df['Generated Docstring'].apply(process_docstring)
    
    # Write the modified DataFrame to a text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for docstring in df['Generated Docstring']:
            f.write(docstring + '\n')
    
    print(f"Processed data written to {output_file}")

# File paths
input_file = 'python_docstring.tsv'
output_file = 'python_docstring_postprocessed.txt'

# Process the TSV file and write to text file
process_tsv(input_file, output_file)
