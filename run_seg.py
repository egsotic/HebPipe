import multiprocessing as mp
import glob
import sys
import argparse
import os
import tqdm
    
sys.path.append('/home/nlp/egsotic/repo/HebPipe/hebpipe/')

from lib.whitespace_tokenize import tokenize as whitespace_tokenize
from rftokenizer import RFTokenizer


DATA_DIR = '/home/nlp/egsotic/repo/HebPipe/hebpipe/data/'

MORPH_DELIMITER = '‡'

import tokenizers
tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()

def tokenize_text(text):
    return [w for w, _ in tokenizer.pre_tokenize_str(text)]

def batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]

def process(input_path, output_path):
    my_tokenizer = RFTokenizer(model="heb")
    
    total_lines = count_file_lines(input_path)
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        all_lines = list(f_in)
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for lines in batch_iter(all_lines, batch_size=256):
                try:
                    # text = line.strip()
                    # tokenized_text = whitespace_tokenize(text, abbr=DATA_DIR + "heb_abbr.tab", add_sents=False, from_pipes=False)
                    # tokenized_segmented_text = my_tokenizer.rf_tokenize(tokenized_text.strip().split("\n"))
                    tokenized_text = sum(([';;;'] + tokenize_text(line) for line in lines), [])[1:]
                    tokenized_segmented_text = my_tokenizer.rf_tokenize(tokenized_text)
                    segmented_text = " ".join(tokenized_segmented_text).replace(';;;', '\n')
                    # segmented_text = segmented_text.replace('|', MORPH_DELIMITER)
                    
                    f_out.write(segmented_text + os.linesep)
                    f_out.flush()
                except Exception as e:
                    print(e)

def process_parallel_worker(args):
    input_path, output_path = args
    
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    
    process(input_path, output_path)

def get_files(glob_input_path, pattern_output_path):
    source, target = pattern_output_path.split(' ')
    
    for file_input_path in glob.glob(glob_input_path):
        file_output_path = file_input_path.replace(f'/{source}/', f'/{target}/')
        
        print(file_output_path)
        
        if not os.path.exists(file_output_path):
            print(file_input_path, '->', file_output_path)
            
            os.makedirs(os.path.dirname(file_output_path), exist_ok=True)
            
            yield file_input_path, file_output_path

def count_file_lines(path):
    with open(path, 'r', encoding='utf-8') as f_in:
        total_lines = sum(1 for _ in f_in)
    
    return total_lines


def process_parallel(glob_input_path, pattern_output_path):
    print(len(glob.glob(glob_input_path)), 'files')
    print(sum(1 for _ in get_files(glob_input_path, pattern_output_path)), 'files')
    
    with mp.Pool(10) as pool:
        for _ in pool.imap_unordered(process_parallel_worker,
                                     get_files(glob_input_path, pattern_output_path)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--parallel", action="store_true")
    
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    parallel = args.parallel
    
    if parallel:
        process_parallel(input_path, output_path)
    else:
        process(input_path, output_path)
