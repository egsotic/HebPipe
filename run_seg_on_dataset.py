import os
import re
import json
import tqdm
import conllu


def ner_dataset_reader(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        sent = {
            'tokens': [],
            'labels': [],
        }
        
        for line in f:
            line = line.strip()
            if line == '':
                yield sent
                
                sent = {
                    'tokens': [],
                    'labels': [],
                }
            else:
                token, label = line.strip().rsplit(maxsplit=1)
                sent['tokens'].append(token)
                sent['labels'].append(label)
        
    if len(sent['tokens']) > 0:
        yield sent

def ner_dataset_writer(sent_iter, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in sent_iter:
            for token, label in zip(sent['prepared_seg_tokens'], sent['labels']):
                f.write(f'{token} {label}{os.linesep}')
            f.write(os.linesep)
            f.flush()

def qa_dataset_reader(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def qa_dataset_processor(dataset, sent_tokenize, tokenize, prepare_for_tokenizer, tokenize_kwargs):
    new_dataset = {
        k: v
        for k, v in dataset.items()
        if k != 'data'
    }
    new_dataset['data'] = []
    
    tokenizer_delimiter = tokenize_kwargs['tokenizer_delimiter']
    # try matching delimiter as prefix or suffix, therefore 0 to 2 times
    sep_regex = f'{re.escape(tokenizer_delimiter)}' + '{0,2}'
    
    for d in tqdm.tqdm(dataset['data']):
        seg_context = prepare_for_tokenizer(' '.join(tokenize(sent_tokenize(d['context']))))
        seg_question = prepare_for_tokenizer(' '.join(tokenize(sent_tokenize(d['question']))))

        answer = prepare_for_tokenizer(' '.join(sent_tokenize(d['answers']['text'][0])))
        
        seg_answer_regex = re.compile('<R>'.join(re.escape(c) for c in answer).replace('<R>', sep_regex))
        seg_answers = {'text': [], 'answer_start': []}
        
        try:
            for m in seg_answer_regex.finditer(seg_context):
                seg_answers['text'].append(m.group())
                seg_answers['answer_start'].append(m.start())
        except:
            print(seg_context)
            raise
        
        if len(seg_answers) == 0:
            print("conversion failed: could not find answer in segmented context")
            print(d)
            print('seg_context', seg_context)
            continue
        
        new_d = d.copy()
        new_d['context'] = seg_context
        new_d['question'] = seg_question
        new_d['answers'] = seg_answers

        new_dataset['data'].append(new_d)
    
    return new_dataset

def qa_dataset_writer(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

def read_conllu(data_path):
    with open(data_path) as f:
        for tl in conllu.parse_incr(f):
            yield tl

def token_morph_iter(tl):
    i = 0
    while i < len(tl):
        t = tl[i]
        t_i = i

        # multi
        if isinstance(t['id'], tuple):
            start_i, _, end_i = t['id']
            n_morphs = (end_i - start_i) + 1
            morphs = tl[i + 1: i + 1 + n_morphs]
            i += n_morphs + 1
        else:
            morphs = []
            i += 1

        yield t_i, t, morphs

def conllu_dataset_reader(input_path):
    yield from read_conllu(input_path)

def conllu_dataset_processor(dataset, sent_tokenize, tokenize, prepare_for_tokenizer, tokenize_kwargs):
    for tl in dataset:
        tokens_morphs = [(t, ms) for _, t, ms in token_morph_iter(tl)]
        tokens = [t['form'] for t, _ in tokens_morphs]
        
        seg_tokens = tokenize(tokens)
        prepared_tokens = [prepare_for_tokenizer(t) for t in seg_tokens]
        
        new_tl = []
        for prepared_token, (t, ms) in zip(prepared_tokens, tokens_morphs):
            new_t = t.copy()
            new_t['form'] = prepared_token
            new_tl.extend([new_t] + ms)
        
        new_tl = conllu.TokenList(new_tl, metadata=tl.metadata)
        
        yield new_tl

def conllu_dataset_writer(dataset, output_path):
    save_conllu(tqdm.tqdm(dataset), output_path)
    
def save_conllu(data, path):
    with open(path, 'w') as f:
        for tl in data:
            f.write(serialize_tl(tl))
            
def serialize_tl(tl, tl_fields_order=conllu.parser.DEFAULT_FIELDS):
    tl_ordered = conllu.models.TokenList(
        [reorder_dict(t, tl_fields_order) for t in tl],
        metadata=tl.metadata
    )
    
    return conllu.serializer.serialize(tl_ordered)

def reorder_dict(d, keys):
    return {
        k: d.get(k, None)
        for k in keys
    }

def seg_sent(sent_iter, tokenize):
    for sent in sent_iter:
        sent['seg_tokens'] = tokenize(sent['tokens'])
        
        yield sent


def tqdm_wrap(iterable, **kwargs):
    with tqdm.tqdm(**kwargs) as pbar:
        for x in iterable:
            yield x
            pbar.update()

def get_regex_for_prepare_for_tokenizer(prefix_sep: str, suffix_sep: str):
    prefix_sep_regex = []
    suffix_sep_regex = []
    
    if prefix_sep:
        prefix_sep_regex = [fr"(?:(?<=[משהוכלב]){re.escape(prefix_sep)})"]
    
    if suffix_sep:
        suffix_sep_regex = [fr"(?:{re.escape(suffix_sep)}(?=[יךהוםןכנ]))"]
    
    preseg_sep_regex = re.compile('|'.join(prefix_sep_regex + suffix_sep_regex))
    
    return preseg_sep_regex

def get_prepare_for_tokenizer_func(prefix_sep: str, suffix_sep: str, tokenizer_delimiter: str = 'o'):
    sep_regex = get_regex_for_prepare_for_tokenizer(prefix_sep, suffix_sep)
    
    return lambda text: re.sub(sep_regex, tokenizer_delimiter, text)

def prepare_seg_sent_for_tokenizer(sent_iter, prefix_sep: str, suffix_sep: str, tokenizer_delimiter: str='o'):
    sep_regex = get_regex_for_prepare_for_tokenizer(prefix_sep, suffix_sep, tokenizer_delimiter)
                 
    for sent in sent_iter:        
        prepared_tokens = [
            re.sub(sep_regex, tokenizer_delimiter, t)
            for t in sent['seg_tokens']
        ]
        
        sent['prepared_seg_tokens'] = prepared_tokens
        
        yield sent

def get_sent_tokenize(name, kwargs):
    if name == 'bert_pre_tok':
        import tokenizers
        tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()

        def tokenize_text(text):
            return [w for w, _ in tokenizer.pre_tokenize_str(text)]

        return tokenize_text
    
    raise Exception('unknown sent tokenize name', name)

def get_tokenize(name, kwargs):
    if name == "rft":
        from rftokenizer import RFTokenizer
        
        rf_tokenizer = RFTokenizer(model="heb")
        rf_tokenizer.load(None if kwargs['rft_model_path'] is None else os.path.join(kwargs['rft_model_path'], 'heb.sm3'))
        
        return rf_tokenizer.rf_tokenize
    
    elif name == "presuf":
        import sys
        sys.path.append('/home/nlp/egsotic/repo/academic-budget-bert')
        from presuf_tokenize import heb_separate_prefixes_suffixes
        
        def tokenize(tokens):
            return [
                heb_separate_prefixes_suffixes(t)
                for t in tokens
            ]
            
        return tokenize
    
    raise Exception('unknown tokenize name', name)

def get_dataset_pipeline(dataset_type):
    if dataset_type == 'qa':
        return qa_dataset_reader, qa_dataset_processor, qa_dataset_writer
    
    elif dataset_type == 'ner':
        raise Exception('fix code')
    
    elif dataset_type == 'conllu':
        return conllu_dataset_reader, conllu_dataset_processor, conllu_dataset_writer
    
    raise Exception('unknown dataset name', dataset_type)
    
def main(inputs_path, get_output_path, dataset_type, tokenize_kwargs):
    prefix_sep = tokenize_kwargs['prefix_sep']
    suffix_sep = tokenize_kwargs['suffix_sep']
    tokenizer_delimiter = tokenize_kwargs['tokenizer_delimiter']
    
    sent_tokenize = get_sent_tokenize(tokenize_kwargs['sent_tokenize_type'], tokenize_kwargs)
    tokenize = get_tokenize(tokenize_kwargs['tokenize_type'], tokenize_kwargs)
    prepare_for_tokenizer = get_prepare_for_tokenizer_func(prefix_sep,
                                                           suffix_sep,
                                                           tokenizer_delimiter)
    
    dataset_reader, dataset_processor, dataset_writer = get_dataset_pipeline(dataset_type)
    
    for input_path in inputs_path:
        output_path = get_output_path(input_path)
        
        print(input_path, '->', output_path)
        
        dataset_writer(
            dataset_processor(
                dataset_reader(
                    input_path
                ),
                sent_tokenize,
                tokenize,
                prepare_for_tokenizer,
                tokenize_kwargs,
            ),
            output_path,
        )
        
        # sent_iter = dataset_reader(input_path)
        # seg_sent_iter = seg_sent(sent_iter, tokenize)
        # prepared_sent_iter = prepare_seg_sent_for_tokenizer(seg_sent_iter,
        #                                                     prefix_sep=tokenize_kwargs['prefix_sep'],
        #                                                     suffix_sep=tokenize_kwargs['suffix_sep'],
        #                                                     tokenizer_delimiter=tokenize_kwargs['tokenizer_delimiter'])
        # tqdm_prepared_sent_iter = tqdm_wrap(prepared_sent_iter, desc='sent')
        # dataset_writer(tqdm_prepared_sent_iter, output_path)

if __name__ == "__main__":
    # get_dataset_input_path = lambda split: f'/home/nlp/egsotic/data/qa/parashoot/v1_original/{split}.json'
    # output_dir = '/home/nlp/egsotic/data/qa/parashoot/v1_original_preseg_rft/'
    # get_dataset_input_path = lambda split: f'/home/nlp/egsotic/data/ud-treebanks-v2.11/UD_Hebrew-HTB_nemo/he_htb_nemo-ud-{split}.conllu'
    get_dataset_input_path = lambda split: f'/home/nlp/egsotic/data/ud-treebanks-v2.11/UD_Hebrew-IAHLTwiki/he_iahltwiki-ud-{split}.conllu'
    # rft_model = 'iahltwiki'
    # rft_model = 'htb'
    # output_dir = f'/home/nlp/egsotic/data/ud-treebanks-v2.11/UD_Hebrew-HTB_nemo.preseg_rft_{rft_model}'
    # output_dir = f'/home/nlp/egsotic/data/ud-treebanks-v2.11/UD_Hebrew-IAHLTwiki.preseg_rft_{rft_model}'
    output_dir = f'/home/nlp/egsotic/data/ud-treebanks-v2.11/UD_Hebrew-IAHLTwiki.preseg_presuf'
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_type = 'conllu'
    
    # default
    # get_output_path = lambda input_path: os.path.join(output_dir, os.path.basename(input_path))
    
    # conllu
    get_split = lambda input_path: os.path.basename(input_path).rsplit('-', maxsplit=1)[1].split('.')[0]
    # get_output_path = lambda input_path: os.path.join(output_dir, f'he_htb_nemo.preseg_rft_{rft_model}-ud-{get_split(input_path)}.conllu')
    # get_output_path = lambda input_path: os.path.join(output_dir, f'he_iahltwiki.preseg_rft_{rft_model}-ud-{get_split(input_path)}.conllu')
    get_output_path = lambda input_path: os.path.join(output_dir, f'he_iahltwiki.preseg_presuf-ud-{get_split(input_path)}.conllu')
    
    inputs_path = [
        get_dataset_input_path(split)
        for split in ['dev', 'test','train']
    ]
    
    tokenize_kwargs = {
        'sent_tokenize_type': 'bert_pre_tok',
        'tokenize_type': 'presuf',
        # 'rft_model_path': f'/home/nlp/egsotic/data/RFTokenizer/{rft_model}_only',
        'prefix_sep': '_',
        'suffix_sep': '@',
        'tokenizer_delimiter': 'o',
    }
    
    main(inputs_path, get_output_path, dataset_type, tokenize_kwargs)
