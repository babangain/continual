import argparse
from codebleu import calc_codebleu

def read_file(filepath):
    with open(filepath, 'r') as file:
        return [line.strip() for line in file.readlines()]

def main():
    parser = argparse.ArgumentParser(description='Calculate CodeBLEU score.')
    parser.add_argument('--ref', type=str, help='Path to the reference file')
    parser.add_argument('--hyp', type=str, help='Path to the generated file')
    parser.add_argument('--lang', type=str, default='python', help='Programming language (default: python)')

    args = parser.parse_args()

    references = read_file(args.ref)
    predictions = read_file(args.hyp)

    # Calculate CodeBLEU score
    res = calc_codebleu(references, predictions, args.lang)
    
    print("CodeBLEU score results:")
    print(f"CodeBLEU: {res['codebleu']}")
    print(f"Ngram Match Score: {res['ngram_match_score']}")
    print(f"Weighted Ngram Match Score: {res['weighted_ngram_match_score']}")
    print(f"Syntax Match Score: {res['syntax_match_score']}")
    print(f"Dataflow Match Score: {res['dataflow_match_score']}")

if __name__ == "__main__":
    main()
