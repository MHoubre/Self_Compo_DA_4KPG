from datasets import load_dataset, concatenate_datasets
import re
import argparse

DIGIT_token = "<digit>"
EOS_token = " . <eos> "

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

def add_peos(dataset):
    peos_written=0
    keyphrases=[dataset["keyphrases"][0]]
    #We write the first keyphrase depending on if it's an absent keyphrase or a present keyphrase
    if dataset["prmu"][0]!="P":
        keyphrases.append("<peos>")
        peos_written=1
    
    for i in range(1,len(dataset["prmu"])):
        if peos_written == 0:
            if dataset["prmu"][i] != "P" and dataset["prmu"][i-1] == "P":
                keyphrases.append("<peos>")
                peos_written=1
        if i == len(dataset["prmu"])-1 :
            #print(len(dataset["prmu"])==len(dataset["keyphrases"]))
            keyphrases.append(dataset["keyphrases"][-1])

        else:    
            keyphrases.append(dataset["keyphrases"][i])

    if not peos_written:
        keyphrases.append("<peos>")
    dataset["keyphrases"]=";".join(keyphrases)

    return dataset

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--data_file")
    parser.add_argument("-r","--relation_file",required=False)
    parser.add_argument("-s","--output_src_file")
    parser.add_argument("-t","--output_trg_file")

    args = parser.parse_args()
    
    data = load_dataset("json",data_files=args.data_file)
    data = data["train"]

    # Processing the text

    if args.relation_file != None:
        rel_data = load_dataset("json",data_files=args.relation_file)
        
        rel_data = rel_data.map(lambda ex:{"title": ex["text"].split("<s>")[0].lower()})
        rel_data = rel_data.map(lambda ex:{"abstract": ex["text"].split("<s>")[1].lower()})
        rel_data = rel_data["train"]

        data = concatenate_datasets([data,rel_data])

    data = data.map(lambda ex:{"title":ex["title"].lower()})
    data = data.map(lambda ex:{"abstract":ex["abstract"].lower()})
    data = data.map(meng17_tokenize_column,fn_kwargs={"column":"title"},desc="Meng 17 tokenization on title")
    data = data.map(meng17_tokenize_column,fn_kwargs={"column":"abstract"}, desc="Meng 17 tokenization on abstract")
    data = data.map(lambda ex:{"text":ex["title"]+EOS_token+ex["abstract"]},desc="Getting final input")

    # Processing the keyphrases sequence
    data = data.map(lambda ex:{"keyphrases":[element.lower() for element in ex["keyphrases"]]})
    data = data.map(meng17_tokenize_column,fn_kwargs={"column":"keyphrases"},desc="Meng 17 tokenization on reference keyphrases")
    #data = data.map(lambda ex:{"keyphrases":";".join(ex["keyphrases"])},num_proc=8,desc="Joining into final reference sequence")
    data = data.map(add_peos)
       
    print(data["keyphrases"][0])

    with open(args.output_src_file,"w") as out_src:
        for line in data:
            out_src.write(line["text"])
            out_src.write("\n")

    if args.output_trg_file != None: 
        with open(args.output_trg_file,"w") as out_trg:
            for line in data:
                out_trg.write(line["keyphrases"])
                out_trg.write("\n")
