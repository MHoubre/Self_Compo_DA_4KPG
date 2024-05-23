#!/usr/bin/env python
# coding: utf-8

from this import d
from datasets import load_dataset, load_from_disk, ReadInstruction, concatenate_datasets
import spacy
import re
# from spacy.lang.en import English
from spacy.tokenizer import _get_regex_pattern
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from nltk.stem.porter import PorterStemmer 
import numpy as np
import argparse


stemmer = PorterStemmer()

nlp = spacy.load("en_core_web_sm",disable=['tagger','parser','ner','lemmatizer','textcat'])

# Modify tokenizer infix patterns
infixes = (
    LIST_ELLIPSES
    + LIST_ICONS
    + [
        r"(?<=[0-9])[+\-\*^](?=[0-9-])",
        r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
            al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
        ),
        r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
        # ✅ Commented out regex that splits on hyphens between letters:
        # r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
    ]
)

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

"""
Function that returns if a subseq is in the inseq
"""
def contains(subseq, inseq):
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))


"""
Function that returns the position of a subsequence if it is found in the larger text.
"""
def contains_with_pos(subseq, inseq):
    for pos in range(0, len(inseq) - len(subseq) + 1):
        if inseq[pos:pos + len(subseq)] == subseq:
            return pos

"""
Function that gets all the keyphrases with a P label in prmu column
"""
def get_presents(dataset):
    
    presents = np.where(np.isin(dataset["prmu"], "P"))[0].tolist()
    presents_kp = [dataset["tokenized_keyphrases"][index] for index in presents]
    dataset["presents"]=presents_kp
    return dataset

"""
Function that joins the tokenized present keyphrases
"""
def join_tokenized_present_keyphrases(dataset):
    
    kp_stems = []
    for element in dataset["presents"]:
        element = " ".join(element)
        kp_stems.append(element)
    dataset["presents"] = kp_stems   
    
    return dataset
    

"""
Function that gets the offset for all present keyphrases
"""
def get_present_order(dataset):

    position=[]
    for element in dataset["presents"]:
        position.append(contains_with_pos(element, dataset["tok_text"]))
        #print(position)
    dataset["ordered_present_offset"] = position
    return dataset

"""
Function that orders the present stemmed keyphrases by their offset
returns a dataset with a "ordered_present_kp" column
"""
def reorder_present_kp(dataset):
    if "P" not in dataset["prmu"]:
        cop=[]
    else:
        cop = [x for _, x in sorted(zip(dataset["ordered_present_offset"], dataset["presents"])
                                    , key=lambda pair: pair[0])]
    dataset["ordered_presents"] = cop
    return dataset

"""
Function that orders the present keyphrases and their prmu
"""
def reorder_kp(dataset):
    reordered_kp=[]
    reordered_prmu=[]
    for stem_kp in dataset["ordered_presents"]:

        index = dataset["tokenized_keyphrases"].index(stem_kp)
        reordered_kp.append(dataset["keyphrases"][index])
        reordered_prmu.append(dataset["prmu"][index])
    dataset["reordered_keyphrases"] = reordered_kp
    dataset["reordered_prmu"] = reordered_prmu
    return dataset

def add_absent_kp(dataset):
    kps = dataset["reordered_keyphrases"]
    prmu = dataset["reordered_prmu"] 
    for i,kp in enumerate(dataset["keyphrases"]):
        if kp not in dataset["reordered_keyphrases"]:
            kps.append(kp)
            prmu.append(dataset["prmu"][i])
    dataset["reordered_prmu"] = prmu
    dataset["reordered_keyphrases"] = kps
    return dataset
    

def find_prmu(tok_title, tok_abstract,tok_kp):
    """Find PRMU category of a given keyphrase."""

    # if kp is present
    if contains(tok_kp, tok_title) or contains(tok_kp,tok_abstract):
        return "P"

    # if kp is considered as absent
    else:

        # find present and absent words
        present_words = [w for w in tok_kp if w in tok_title ]

        # if "all" words are present
        if len(present_words) == len(tok_kp):
            return "R"
        # if "some" words are present
        elif len(present_words) > 0:
            return "M"
        # if "no" words are present
        else:
            return "U"
    return prmu

def tokenize_and_stemm_column(dataset,column):
    keyphrases_stems= []
    for keyphrase in dataset[column]:
        keyphrase_spacy = nlp(keyphrase)
        keyphrase_tokens = [token.text for token in keyphrase_spacy]
        keyphrase_stems = [PorterStemmer().stem(w.lower()) for w in keyphrase_tokens]
        keyphrases_stems.append(keyphrase_stems)
        
    dataset["tokenized_{}".format(column)] = keyphrases_stems
    return dataset

def tokenize_and_stemm_text(dataset):

    title = dataset["title"]#.split("<s>")[0]
    abstract = dataset["abstract"]#.split("<s>")[1]
    title_spacy = nlp(title)
    abstract_spacy = nlp(abstract)
    #abstract_spacy = nlp(dataset['abstract'])

    title_tokens = [token.text for token in title_spacy]
    abstract_tokens = [token.text for token in abstract_spacy]
    #abstract_tokens = [token.text for token in abstract_spacy]

    title_stems = [PorterStemmer().stem(w.lower()) for w in title_tokens]
    abstract_stems = [PorterStemmer().stem(w.lower()) for w in title_tokens]
    #abstract_stems = [PorterStemmer().stem(w.lower()) for w in abstract_tokens]

    dataset["title_stems"] = title_stems
    dataset["abstract_stems"] = abstract_stems
    #dataset["abstract_stems"] = abstract_stems
    return dataset

"""
Function that tokenizes the dataset (title, text and keyphrases)
and runs the prmu algorithm.
"""
def prmu_dataset(dataset,column):

    title_stems = dataset["title_stems"]
    abstract_stems = dataset["abstract_stems"]
    #abstract_stems = dataset["abstract_stems"]
    prmu = [find_prmu(title_stems,abstract_stems, kp) for kp in dataset["tokenized_{}".format(column)]]

    dataset['prmu_{}'.format(column)] = prmu

    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description='prmu statistics',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d","--data_file")
    parser.add_argument("-o","--output_file")
    parser.add_argument("-r","--reorder")
    parser.add_argument("-p","--predictions_file")

    args = parser.parse_args()

    dataset = load_dataset("json",data_files=args.data_file)
    dataset = dataset["train"]

    to_keep = ["name","title","abstract","keyphrases","top_m","top_5","top_10"]

    #dataset = dataset.rename_column("label","keyphrases")
    dataset = dataset.map(lambda ex:{"keyphrases":ex["keywords"].split(";")})

    if args.predictions_file != None:
        predictions_data = load_dataset("json",data_files=args.predictions_file)
        for column in list(predictions_data["train"].features.keys()):
            print(column)
            print(predictions_data["train"][0][column])
            if column not in list(dataset.features.keys()):
                dataset = dataset.add_column(column,predictions_data["train"][column])
        columns = ["top_m","top_5","top_10"]
    else:
        columns = ["keyphrases"]
        

    dataset = dataset.map(tokenize_and_stemm_text) # We do he same to the text for comparison
    

    print("PRMU")
    for column in columns:
        to_keep.append("prmu_{}".format(column))
        dataset = dataset.map(tokenize_and_stemm_column,fn_kwargs={"column": column}) # We need to tokenize and stemm to get PRMU labels
        dataset = dataset.map(prmu_dataset,fn_kwargs={"column": column}) # Getting the PRMU labels for each keyphrase

    #dataset["train"].to_json("data_prmu.jsonl")

    if args.reorder:

        print("GETTING PRESENTS")
        dataset = dataset.map(get_presents, num_proc=8)

        print("GETTING ORDER")
        dataset = dataset.map(get_present_order,num_proc=8)

        print("REORDERING")
        dataset = dataset.map(reorder_present_kp,num_proc=8) # Reordering the present keyphrases by their occurrence in the source text

        dataset = dataset.remove_columns(["title_stems","abstract_stems","tokenized_{}".format(args.column)])

        dataset = dataset.map(reorder_kp,num_proc=8)

        dataset = dataset.map(add_absent_kp,num_proc=8)

    dataset = dataset.remove_columns([column for column in list(dataset.features.keys()) if column not in to_keep])

    dataset.to_json(args.output_file)