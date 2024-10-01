from argparse import ArgumentParser
from this import d
from datasets import load_dataset, load_from_disk, ReadInstruction
import spacy
import re
# from spacy.lang.en import English
from spacy.tokenizer import _get_regex_pattern
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from nltk.stem.snowball import SnowballStemmer as Stemmer
import numpy as np

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

def tokenize_and_stemm_keyphrases(dataset):
    keyphrases_stems= []
    for keyphrase in dataset["keyphrases"]:
        keyphrase_spacy = nlp(keyphrase)
        keyphrase_tokens = [token.text for token in keyphrase_spacy]
        keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
        keyphrases_stems.append(keyphrase_stems)
        
    dataset["tokenized_keyphrases"] = keyphrases_stems
    return dataset

def tokenize_and_stemm_text(dataset):
    title_spacy = nlp(dataset['title'])
    abstract_spacy = nlp(dataset['abstract'])

    title_tokens = [token.text for token in title_spacy]
    abstract_tokens = [token.text for token in abstract_spacy]

    title_tokens[-1] = title_tokens[-1]+"."

    title_stems = [Stemmer('porter').stem(w.lower()) for w in title_tokens]
    abstract_stems = [Stemmer('porter').stem(w.lower()) for w in abstract_tokens]

    dataset["title_stems"] = title_stems
    dataset["abstract_stems"] = abstract_stems
    return dataset

"""
Function that tokenizes the dataset (title, text and keyphrases)
and runs the prmu algorithm.
"""
def prmu_dataset(dataset):

    title_stems = dataset["title_stems"]
    abstract_stems = dataset["abstract_stems"]
    prmu = [find_prmu(title_stems, abstract_stems, kp) for kp in dataset["tokenized_keyphrases"]]

    dataset["tok_text"] = title_stems + abstract_stems
    dataset['prmu'] = prmu

    return dataset



if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-i","--input_file",help="Input jsonl file which for each line has a title, an abstract and keyphrases")
    parser.add_argument("-o","--output_file", help="Output file")
    parser.add_argument("-r","--reorder",action='store_true',help="If you want your keyphrases to be reordered with the presents in their apparition order in the text and the absents after")

    args = parser.parse_args()

    dataset = load_dataset("json",data_files=args.input_file)

    print("TOKENIZATION")
    dataset = dataset.map(tokenize_and_stemm_keyphrases, desc="Pre-processing keyphrases") # We need to tokenize and stemm to get PRMU labels

    dataset = dataset.map(tokenize_and_stemm_text, desc="Pre-processing text") # We do he same to the text for comparison
    

    print("PRMU")
    dataset = dataset.map(prmu_dataset, desc="Assigning a PRMU label to keyphrases") # Getting the PRMU labels for each keyphrase



    #dataset["train"].to_json("data_prmu.jsonl")

    print("GETTING PRESENTS")
    dataset = dataset.map(get_presents, num_proc=8, desc="Getting which keyphrases are present ones")

    print("GETTING ORDER")
    dataset = dataset.map(get_present_order,num_proc=8, desc="Getting the apparition order of present keyphrases")


    if args.reorder:
        print("REORDERING")
        dataset = dataset.map(reorder_present_kp,num_proc=8, desc="Reordering the stemmed present keyphrases by their occurrence in the source text")

        dataset = dataset.map(reorder_kp,num_proc=8, desc="Reordering full present keyphrases")

        dataset = dataset.map(add_absent_kp,num_proc=8, desc="Adding the absent keyphrases in their original order")

        dataset = dataset.remove_columns(["abstract_stems","title_stems","tok_text"])


    dataset['train'].to_json(args.output_file)
