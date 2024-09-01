import csv
import re

def extract_docstrings(file_name):
    docstring_pattern = re.compile(r"(['\"]{3})(.*)")

    with open(file_name, 'r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        i = 0
        
        for row in reader:
            i += 1
            if i == 20:
                break
            print(i, end="\t\t\t")
            docstring = row.get('Generated Docstring', '')
            if docstring.startswith('Code:'):

                match = docstring_pattern.search(docstring)
                if match:
                    print(match.group(2).strip())
                else:
                    #print(docstring.strip())
                    print(" ")
            else:
                print(docstring.strip())

# Call the function with the input file name
extract_docstrings('continual/outputs/baseline/code2text/python_docstring.tsv')
