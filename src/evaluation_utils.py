from this import d
from datasets import load_dataset, load_from_disk, ReadInstruction
import spacy
import re
# from spacy.lang.en import English
from spacy.tokenizer import _get_regex_pattern
from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS, HYPHENS
from spacy.util import compile_infix_regex
from nltk.stem.snowball import SnowballStemmer as Stemmer
import numpy as np


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
        r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
        r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        r"(?<=[0-9])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS)
    ]
)

nlp = spacy.load("en_core_web_sm",disable=['tagger','parser','ner','lemmatizer','textcat'])

infix_re = compile_infix_regex(infixes)
nlp.tokenizer.infix_finditer = infix_re.finditer

def tokenize_and_stemm_keyphrases(dataset):
    keyphrases_stems= []
    for keyphrase in dataset["keyphrases"].split(";"):
        keyphrase_spacy = nlp(keyphrase)
        keyphrase_tokens = [token.text for token in keyphrase_spacy]
        keyphrase_stems = " ".join([Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens])
        #print(keyphrase_stems)
        keyphrases_stems.append(keyphrase_stems)
        
    dataset["tokenized_keyphrases"] = keyphrases_stems
    return dataset

"""
Function that splits the sequence of keyword that the model gives us
"""
def predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]: #for each sequence in predictions
        seq = re.sub(r'<unk>|<s>|<\/s>|<pad>|<\/unk>','',seq) #We get rid of residual special tokens
        seq = [re.sub(r'^ | $','',sp) for sp in seq.split(";")]
        #print(seq)
        if seq[-1]=='':
            seq = seq[:-1] #the split with <KP> leaves residual whitespaces at the beginning or and ending of keyphrases
        splits.append(seq) 
    dataset["pred"] = splits
    dataset["splits"] = splits

    return dataset

"""
Function that splits the sequence of keyword that the model gives us
"""
def reconstruction_predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]: #for each sequence in predictions
        seq = re.sub(r'<unk>|<s>|<\/s>|<pad>|<\/unk>| <KP> | <\/KP> | <digit>','',seq) #We get rid of residual special tokens
        seq = re.sub(r'<digit>','',seq) #We get rid of residual special tokens
        seq = [re.sub(r'^ | $','',sp) for sp in seq.split(";")]
        #print(seq)
        if seq[-1]=='':
            seq = seq[:-1] #the split with <KP> leaves residual whitespaces at the beginning or and ending of keyphrases
        splits.append(seq) 
    dataset["pred"] = splits
    dataset["splits"] = splits

    return dataset

def remove_unk(dataset, n):
    dataset["top_{}".format(n)] = [kp for kp in dataset["top_{}".format(n)] if kp !="<unk>"]
    return dataset

def chowdhury_predictions_split(dataset):
    splits = []
    for seq in dataset["pred"]:
        seq = re.sub(r'\s{2}|<$|<K$|<KP$|\([a-zA-Z]+$','',seq)        
        splits.append(seq.split(","))
    dataset["splits"] = splits

    return dataset

def get_presents(dataset):
    presents = np.where(np.isin(dataset["prmu"], "P"))[0].tolist()
    presents = [dataset["tokenized_keyphrases"][i] for i in presents] #Getting tokenized present keyphrases for evaluation
    dataset["presents"]=presents
    return dataset


def get_absents(dataset):
    absents = np.where(np.isin(dataset["prmu"], "P", invert=True))[0].tolist()
    absents = [dataset["tokenized_keyphrases"][i] for i in absents] #Getting tokenized absent keyphrases for evaluation
    dataset["absents"] = absents
    return dataset

def tokenize(kp):
    keyphrase_tokens = kp.split()
    keyphrase_stems = [Stemmer('porter').stem(w.lower()) for w in keyphrase_tokens]
    return " ".join(keyphrase_stems)
  

def tokenize_keyphrases(dataset):
    keyphrases = []
    for keyphrase in dataset["keyphrases"]:
        keyphrases.append(tokenize(keyphrase))
    dataset["tokenized_keyphrases"] = keyphrases
    return dataset
    
"""
Function that tokenizes the keyphrases before looking for the topk
""" 
def tokenize_predictions(dataset):
    tok_preds=[]
    for kp_list in dataset["pred"]:
        kp_l = []
        for kp in kp_list:
            kp = tokenize(kp)
            kp_l.append(kp)
            

        tok_preds.append(kp_l)
    dataset["splits"] = tok_preds
    return dataset
        

def macro_f1(dataset):

    return np.mean(dataset["test"]["f1_measure"])

def macro_recall(dataset):
    return np.mean(dataset["test"]["recall"])