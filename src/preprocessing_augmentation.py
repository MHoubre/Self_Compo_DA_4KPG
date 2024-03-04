import json
import torch
from nltk.stem.snowball import SnowballStemmer as Stemmer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import time
import numpy as np
from collections import Counter
from data_processing_utils import *
import sys
import json
import io
import random
from multiprocess import set_start_method
from collections import Counter
import os
from timeit import default_timer as timer   

#set_start_method("spawn")


if __name__=="__main__":

    split = sys.argv[1]
    file_path = "improving_from_relation/data/train_{}.jsonl".format(split)

    ids2abstract = {}

    data = load_dataset("json",data_files={split: file_path},cache_dir="/gpfsdswork/projects/rech/rgh/udr36oj")  #Loading the datafile as a Huggingface Dataset

    #data = data.map(meng17_tokenize_column, fn_kwargs={"column":"title"},num_proc=1) #We tokenize the title
    #data = data.map(meng17_tokenize_column, fn_kwargs={"column":"abstract"},num_proc=1) #We tokenize the abstract

    print("GETTING ID2TEXT DICT")

    # It is easier and faster to manipulate a dict of ids to get the abstract instead of using a Dataset object

    for i,row in enumerate(tqdm(data[split])): 
        ids2abstract[row["id"]] = row["abstract"] 

    '''----------------------- Getting keyphrases dict  ------------------------------------'''
    data = data.map(lower_keyphrases, fn_kwargs={"kps_category":"keyphrases"}, num_proc=1) #Lowering all the keyphrases to avoid duplicates due to case sensitive processing

    uniq_kps = set([kp for kp_list in data[split]["lowered_keyphrases"] for kp in kp_list]) # Set of all the different keyphrases in the corpus.

    kp2doc = dict.fromkeys(uniq_kps,[]) #Putting the unique keyphrases as keys of a dictionnary. The value is going to be a list of all the documents that have this keyphrase in their reference

    print("FILLIN THE LINKED KPS DICT")

    kp2doc = fill_kp2doc(kp_lists= data[split]["lowered_keyphrases"], ids=data[split]["id"], kp2doc=kp2doc) #For each keyphrase, we list the documents that have it in their reference

    #doc2kp = {data[split][i]["id"] : data[split]["lowered_keyphrases"] for i,_ in enumerate(tqdm(data[split]))}

    print("GETTING THE DOCS IDs")
    data = data.map(get_linked_documents_ids, fn_kwargs={"kp2doc":kp2doc},num_proc=1)

    uniq_docs = set(data[split]["id"])
    doc2kp = dict.fromkeys(uniq_docs,[])
    doc2kp = fill_doc2kp(doc_list = data[split]["id"],kps=data[split]["lowered_keyphrases"], doc2kp=doc2kp)
    #print(doc2kp['3VPq1pc'])

    data = data.map(get_common_keyphrases_pairs, fn_kwargs={"doc2kp":doc2kp, "n":int(sys.argv[2])},num_proc=1)
    data = data.map(prepare_augmentation_inputs, fn_kwargs={"ids2abstract" : ids2abstract})


    inputs = [element for input_list in data[split]["inputs"] for element in input_list]
    print("NUMBER OF PAIR: {}".format(len(inputs)))

    random.shuffle(inputs) #Shuffling inputs

    with open("improving_from_relation/data/train_{}_{}common_keyphrases.jsonl".format(split,sys.argv[2]),"a") as output:
        for my_input in inputs:

            json.dump({"text":my_input[0],
                    "label":my_input[1]}, output)
            output.write("\n")
