import os
import shutil
import sys
from datasets import load_dataset
from data_processing_utils import meng17_tokenize_column

#in_dir = sys.argv[1]
#out_dir = sys.argv[1]


def get_input(dataset):
    """
    Function that gets the input as a sequence with two subsequences (separated by the BART tokenizer's separator token)
    @param: The Dataset object you are working on
    @return: The Dataset object you are working on with an additional column
    """

    dataset["text"] = dataset["title"] + "<s>" + dataset["abstract"]

    return dataset

def join_keyphrases(dataset):
    dataset["keyphrases"] = ";".join(dataset["keyphrases"])  
    return dataset      



for split in ['train_kp20k']:
    path = os.path.join("improving_from_relation/data/{}.jsonl".format())
    data = load_dataset("json", data_files={split:path})
    #data = data.map(meng17_tokenize_column, fn_kwargs={"column":"title"},num_proc=8) #We tokenize the title
    #data = data.map(meng17_tokenize_column, fn_kwargs={"column":"abstract"},num_proc=8) #We tokenize the abstract
    #data = data.map(meng17_tokenize_column, fn_kwargs={"column":"keyphrases"},num_proc=8) #We tokenize the abstract
    data = data.map(get_input, num_proc=8) # We get the input text at the right format (title + abstract)
    data = data.map(join_keyphrases,num_proc=8) # We put the list of keyphrases in a single sequence
    
    data[split].to_json("improving_from_relation/data/{}_generation.jsonl".format(split))

    # print("getting the source file")
    # out_path = os.path.join(out_dir, split+ "_generation" + '.' + "source")
    # texts = [t for t in data[split]["text"]]
    # with open(out_path, 'w') as f:
    #     f.write('\n'.join(texts))

    # print("getting the target file")
    # out_path = os.path.join(out_dir, split + "_generation" + '.' + "target")
    # keyphrases = [keyphrase_list for keyphrase_list in data[split]["keyphrases"]]
    # with open(out_path, 'w') as f:
    #     f.write('\n'.join(keyphrases))
