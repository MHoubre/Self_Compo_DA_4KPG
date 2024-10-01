import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
from data_processing_utils import lower_keyphrases, fill_kp2doc, fill_doc2kp, get_linked_documents_ids, get_common_keyphrases_pairs, prepare_augmentation_inputs
import json
import random
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser(
    description='prmu statistics',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d","--data_file")
    parser.add_argument("-o","--output_file")

    args = parser.parse_args()

    file_path = args.data_file

    ids2abstract = {}

    data = load_dataset("json",data_files={"train": file_path},cache_dir="/gpfsdswork/projects/rech/rgh/udr36oj")  #Loading the datafile as a Huggingface Dataset


    print("GETTING ID2TEXT DICT")

    # It is easier and faster to manipulate a dict of ids to get the abstract instead of using a Dataset object

    for i,row in enumerate(tqdm(data["train"])): 
        ids2abstract[row["id"]] = row["abstract"] 

    '''----------------------- Getting keyphrases dict  ------------------------------------'''
    data = data.map(lower_keyphrases, fn_kwargs={"kps_category":"keyphrases"}, num_proc=1) #Lowering all the keyphrases to avoid duplicates due to case sensitive processing

    uniq_kps = set([kp for kp_list in data["train"]["lowered_keyphrases"] for kp in kp_list]) # Set of all the different keyphrases in the corpus.

    kp2doc = dict.fromkeys(uniq_kps,[]) #Putting the unique keyphrases as keys of a dictionnary. The value is going to be a list of all the documents that have this keyphrase in their reference

    print("FILLIN THE LINKED KPS DICT")

    kp2doc = fill_kp2doc(kp_lists= data["train"]["lowered_keyphrases"], ids=data["train"]["id"], kp2doc=kp2doc) #For each keyphrase, we list the documents that have it in their reference

    print("GETTING THE DOCS IDs")
    data = data.map(get_linked_documents_ids, fn_kwargs={"kp2doc":kp2doc},num_proc=1) #Getting the ids of the documents that share keyphrases with each doc

    uniq_docs = set(data["train"]["id"])
    doc2kp = dict.fromkeys(uniq_docs,[])
    doc2kp = fill_doc2kp(doc_list = data["train"]["id"],kps=data["train"]["lowered_keyphrases"], doc2kp=doc2kp)  # It is easier and faster to manipulate a dict of ids to get the keyphrases instead of using a Dataset object

    data = data.map(get_common_keyphrases_pairs, fn_kwargs={"doc2kp":doc2kp, "n":0.6},num_proc=1) # Looking at the pairs and selecting those where the link is strong enough
    data = data.map(prepare_augmentation_inputs, fn_kwargs={"ids2abstract" : ids2abstract}) # Creating the artificial documents


    inputs = [element for input_list in data["train"]["inputs"] for element in input_list]
    print("NUMBER OF PAIR: {}".format(len(inputs)))

    random.shuffle(inputs) #Shuffling inputs

    with open(args.output_file,"a") as output:
        for my_input in inputs:

            json.dump({"text":my_input[0],
                    "keyphrases":my_input[1]}, output)
            output.write("\n")
