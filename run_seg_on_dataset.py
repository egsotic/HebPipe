import glob
import os
from rftokenizer import RFTokenizer
import tqdm


def ner_dataset_reader(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        sent = []
        
        for line in f:
            line = line.strip()
            if line == '':
                yield sent[:]
                sent = []
            else:
                word, label = line.strip().rsplit(maxsplit=1)
                sent.append((word, label))
        
        if len(sent) > 0:
            yield sent
            
def ner_dataset_writer(sent_iter, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in sent_iter:
            for word, label in sent:
                f.write(f'{word} {label}{os.linesep}')
            f.write(os.linesep)    
            f.flush()

def seg_sent(sent_iter, tokenizer):
    for sent in sent_iter:
        tokens = [t[0] for t in sent]
        try:
            seg_tokens = tokenizer.rf_tokenize(tokens)
        except:
            import pdb
            pdb.set_trace()
            
            print(tokens)
            print(tokens)
            
        seg_sent = [(seg_token, *t[1:]) for seg_token, t in zip(seg_tokens, sent)]
        yield seg_sent
    
def tqdm_wrap(iterable, **kwargs):
    with tqdm.tqdm(**kwargs) as pbar:
        for x in iterable:
            yield x
            pbar.update()
            
def main(inputs_path, output_dir):
    rf_tokenizer = RFTokenizer(model="heb")
    
    for input_path in inputs_path:
        file_name = os.path.basename(input_path)
        output_path = os.path.join(output_dir, file_name)
        
        sent_iter = ner_dataset_reader(input_path)
        seg_sent_iter = seg_sent(sent_iter, rf_tokenizer)
        seg_sent_iter = tqdm_wrap(seg_sent_iter, desc='sent')
        ner_dataset_writer(seg_sent_iter, output_path)

if __name__ == "__main__":
    inputs_path = [
        os.path.join("/home/nlp/egsotic/data/ner_heb/NEMO/single", f"{split}.txt")
        for split in ['train'] #['dev', 'test','train']
    ]
    
    output_dir = "/home/nlp/egsotic/data/ner_heb/NEMO/single_preseg_rft_default"
    os.makedirs(output_dir, exist_ok=True)
    
    main(inputs_path, output_dir)
