from nltk.stem.snowball import SnowballStemmer as Stemmer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import time
import numpy as np
from collections import Counter
import networkx as nx

import re
import os
import random

DIGIT_token = "<digit>"

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


def tokenize_string(kp, stemm=False):
    keyphrase_tokens = kp.split()
    if stemm:
        keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
    else:
        keyphrase_stems = [w.lower() for w in keyphrase_tokens]
    return " ".join(keyphrase_stems)
  


def lower_keyphrases(dataset, kps_category):
    '''
    Function that lowers (and stems if needed) the keyphrases of a document
    '''
    keyphrases = []
    for keyphrase in dataset[kps_category]:
        keyphrases.append(keyphrase.lower())
    dataset["lowered_"+ kps_category] = keyphrases
    return dataset

def fill_kp2doc(kp_lists,ids,kp2doc):
    for i, kp_list in enumerate(tqdm(kp_lists)): #for each doc that has a list of keyphrases
        #print(kp_list)
        for kp in kp_list: #For each keyphrase kp of a document
            id_list = kp2doc[kp]
            kp2doc[kp] = [*id_list,ids[i]]

            #print(kp2doc[kp])

    return kp2doc

def fill_doc2kp(doc_list,kps,doc2kp):
    for i, doc in enumerate(tqdm(doc_list)): #for each doc that has a list of keyphrases
        
        doc2kp[doc] = kps[i]

            #print(kp2doc[kp])

    return doc2kp

def fill_linkedkp_dict(kp_lists,kp2kp):
    """
    Function that for each keyphrase, lists its neighbouring keyphrases from each documents that contains it.
    """
    for i, kp_list in enumerate(tqdm(kp_lists)): #for each doc
        for kp in kp_list: #For each kp of this document

            linked_kps = kp2kp[kp] #We get the keyphrase sequence associated to this keyphrase

            #We update the sequence by adding the keyphrases of the document that are not yet in the sequence
            tmp1 = linked_kps 
            tmp2 = [k for k in kp_list if k!=kp]
            linked_kps = [*tmp1,*tmp2]

            kp2kp[kp] = linked_kps
           
    return kp2kp


def get_linked_documents_ids(dataset,kp2doc):
    linked_docs = []
    for kp in dataset["lowered_keyphrases"]: #for each keyphrase of a document
        doc_list = list(set(kp2doc[kp]) - {dataset["id"]} ) #We get the list of linked documents to the keyphrase  #We don't want documents to be self linked
        tmp = linked_docs
        linked_docs = [*tmp,*doc_list] #We keep repetitions because that way we can count how many keyphrases are shared between the document and its neighbours.

    sorted_linked_docs = Counter(linked_docs).most_common()

    dataset["linked_documents"] = [element[0] for element in sorted_linked_docs] 
    return dataset


def get_linked_keyphrases(dataset,kp2kp):
    linked_kps = []
    for kp in dataset["lowered_keyphrases"]:
        kp_list = kp2kp[kp] #We get the list of linked documents to the keyphrase
        #print(kp_list)
        tmp = linked_kps
        linked_kps = [*tmp, *kp_list] #We keep repetitions because that way we can count how many keyphrases are shared between the document and its neighbours.

    sorted_linked_kps = Counter(linked_kps).most_common()

    dataset["linked_kps"] = [element[0] for element in sorted_linked_kps] 
    #dataset["number_of_instances_in_common"] = [element[1] for element in sorted_linked_kps]
    return dataset

def get_common_keyphrases_pairs(dataset, doc2kp,n):

    keyphrases = set(dataset["lowered_keyphrases"])
    #print(keyphrases)

    to_silver = []

    for document in dataset["linked_documents"]:  # For each doc with which the document shares keyphrases
        
        maxi = max(len(doc2kp[document]), len(dataset["lowered_keyphrases"]))
        
        if document != dataset["id"]:
            doc_keyphrases_set = set(doc2kp[document]) 
            #print(doc_keyphrases_set)
            common_keyphrases = list(keyphrases & doc_keyphrases_set) # We get the common keyphrases between the two documents

            if len(common_keyphrases) >= maxi*n:
                to_silver.append((document,";".join(common_keyphrases)))
        #print(to_silver)
    if len(to_silver) >= 5:
        to_silver = to_silver[:5]

    dataset["silver_pairs"] = to_silver
    return dataset

def prepare_augmentation_inputs(dataset, ids2abstract):
    title = dataset["title"]

    inputs = []

    #inputs.append(tuple((title+ "<s>" + dataset["abstract"] , ";".join(dataset["keyphrases"]))))
    for silver in dataset["silver_pairs"]:
        inputs.append(tuple((title+ "<s>" + ids2abstract[silver[0]] , silver[1])))

    dataset["inputs"] = inputs
    return dataset


def contains(subseq, inseq):
    """
    Function that returns if a subseq is in the inseq
    """
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))


def find_prmu(tok_title, tok_text, tok_kp):
    """Find PRMU category of a given keyphrase."""

    # if kp is present
    if contains(tok_kp, tok_title) or contains(tok_kp, tok_text):
        return "P"

    # if kp is considered as absent
    else:

        # find present and absent words
        present_words = [w for w in tok_kp if w in tok_title or w in tok_text]

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

def prmu_dataset(dataset,column):

    title_stems = dataset["title_stems"]
    abstract_stems = dataset["abstract_stems"]
    prmu = [find_prmu(title_stems, abstract_stems, kp) for kp in dataset["stemmed_"+column]]

    #dataset["tok_text"] = title_stems + abstract_stems
    dataset[column+'_prmu'] = prmu

    return dataset
