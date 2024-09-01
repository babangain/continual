def replace_tabs_with_spaces(input_file, output_file=None):
    # Open the input file for reading
    with open(input_file, 'r') as file:
        content = file.read()
    
    # Replace tabs with spaces
    content = content.replace('\t', ' ')
    
    # Determine the output file name
    if output_file is None:
        output_file = input_file
    
    # Open the output file for writing
    with open(output_file, 'w') as file:
        file.write(content)
    
    print(f"Tabs in '{input_file}' have been replaced with spaces and saved to '{output_file}'")

# Example usage
input_filename = 'java_reference.txt'
output_filename = 'java_reference.txt.space'  # You can set this to None to overwrite the original file

replace_tabs_with_spaces(input_filename, output_filename)
