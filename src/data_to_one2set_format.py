from datasets import load_dataset
import re
import argparse

DIGIT_token = "<digit>"
EOS_token = "<eos>"

def meng17_tokenize(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits with <digit>, split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :param text:
    :return: a list of tokens
    '''
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = list(filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\']', text)))

    tokens = replace_numbers_to_DIGIT(tokens)

    return tokens

def meng17_tokenize_column(dataset, column):
    if type(dataset[column]) == list:
        text_list=[]
        for text in dataset[column]:
            text = " ".join(meng17_tokenize(text))
            text_list.append(text)
        dataset[column] = text_list
    else:
        dataset[column] = " ".join(meng17_tokenize(dataset[column]))

    return dataset

def replace_numbers_to_DIGIT(tokens, k=2):
    # replace big numbers (contain more than k digit) with <digit>
    tokens = [w if not re.match('^\d{%d,}$' % k, w) else DIGIT_token for w in tokens]

    return tokens

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--data_file")
    parser.add_argument("-s","--output_src_file")
    parser.add_argument("-t","--output_trg_file")

    args = parser.parse_args()
    
    data = load_dataset("json",data_files=args.data_file,cache_dir=".")

    # Processing the text
    #data = data.map(lambda ex:{"title": ex["text"].split("<s>")[0]})
    #data = data.map(lambda ex:{"abstract": ex["text"].split("<s>")[1]})
    data = data.map(meng17_tokenize_column,fn_kwargs={"column":"title"},desc="Meng 17 tokenization on title")
    data = data.map(meng17_tokenize_column,fn_kwargs={"column":"abstract"}, desc="Meng 17 tokenization on abstract")
    data = data.map(lambda ex:{"text":ex["title"]+EOS_token+ex["abstract"]},desc="Getting final input")

    # Processing the keyphrases sequence
    if not isinstance(data["train"]["keyphrases"][0],list):
        data = data.map(lambda ex:{"keyphrases":ex["keyphrases"].split(";")},desc="splitting reference sequence for meng tokenization")
    else:
        data = data.map(meng17_tokenize_column,fn_kwargs={"column":"keyphrases"},desc="Meng 17 tokenization on reference keyphrases")
        data = data.map(lambda ex:{"keyphrases":";".join(ex["keyphrases"])},desc="Joining into final reference sequence")
        
    print(data["train"]["keyphrases"][0])

    with open(args.output_src_file,"a") as out_src:
        for line in data["train"]:
            out_src.write(line["text"])
            out_src.write("\n")
    
    with open(args.output_trg_file,"a") as out_trg:
        for line in data["train"]:
            out_trg.write(line["keyphrases"])
            out_trg.write("\n")