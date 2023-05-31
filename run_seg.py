import sys
import argparse
import os
import tqdm
    
sys.path.append('/home/nlp/egsotic/repo/HebPipe/hebpipe/')

from lib.whitespace_tokenize import tokenize as whitespace_tokenize
from rftokenizer import RFTokenizer


DATA_DIR = '/home/nlp/egsotic/repo/HebPipe/hebpipe/data/'

MORPH_DELIMITER = '‡'


def process(input_path, output_path):
    my_tokenizer = RFTokenizer(model="heb")
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        total_lines = sum(1 for _ in f_in)

    with open(input_path, 'r', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in tqdm.tqdm(f_in, total=total_lines):
                try:
                    text = line.strip()
                    
                    tokenized_text = whitespace_tokenize(text, abbr=DATA_DIR + "heb_abbr.tab", add_sents=False, from_pipes=False)
                    tokenized_segmented_text = my_tokenizer.rf_tokenize(tokenized_text.strip().split("\n"))
                    segmented_text = " ".join(tokenized_segmented_text)
                    segmented_text = segmented_text.replace('|', MORPH_DELIMITER)
                    
                    f_out.write(segmented_text + os.linesep)
                    f_out.flush()
                except Exception as e:
                    print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    
    process(input_path, output_path)
