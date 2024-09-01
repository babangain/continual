import pandas as pd

# Read the TSV file
df = pd.read_csv('generated_java_code.tsv', sep='\t')

# Function to process the Java code
def process_java_code(code):
    # Convert to string
    code = str(code)
    # Replace newlines with spaces
    code = code.replace('\n', ' ')
    # Remove everything after " C# code:"
    if " C# code:" in code:
        code = code.split(" C# code:")[0]
    # Strip the final string
    return code.strip()

# Apply the function to the relevant columns
df['Java Reference Code'] = df['Java Reference Code'].apply(process_java_code)
df['Generated Java Code'] = df['Generated Java Code'].apply(process_java_code)

# Write the processed columns to text files
with open('java_reference_postprocessed.txt', 'w') as ref_file:
    ref_file.write('\n'.join(df['Java Reference Code']))

with open('java_generated_postprocessed.txt', 'w') as gen_file:
    gen_file.write('\n'.join(df['Generated Java Code']))


import pandas as pd

# Read the TSV file
df = pd.read_csv('generated_cs_code.tsv', sep='\t')

# Function to process the C# code
def process_cs_code(code):
    # Convert to string
    code = str(code)
    # Replace newlines with spaces
    code = code.replace('\n', ' ')
    # Remove everything after "Java code:"
    if "Java code:" in code:
        code = code.split("Java code:")[0]
    # Strip the final string
    return code.strip()

# Apply the function to the relevant column
df['Generated C# Code'] = df['Generated C# Code'].apply(process_cs_code)

# Write the processed column to a text file
with open('cs_generated_postprocessed.txt', 'w') as cs_file:
    cs_file.write('\n'.join(df['Generated C# Code']))
