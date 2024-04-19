import json
from tqdm import tqdm
import argparse
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from collections import defaultdict
import numpy as np
from tqdm import tqdm

stemmer = PorterStemmer()
PAD_PHRASE = "<<bad-bad>>"
PAD_MIN = 10

parser = argparse.ArgumentParser()

parser.add_argument("-r",
                    "--reference")

parser.add_argument("-s",
                    "--system")

parser.add_argument("-p","--predictions_type")

parser.add_argument('--output_scores', 
                    )

args = parser.parse_args()


def get_presents(doc):

    presents = np.isin(doc["prmu"], "P").nonzero()[0].tolist() #indexes of the keyphrases that are present in the source text

    return presents

def tokenize(s):
    """Tokenize an input text."""
    return word_tokenize(s)

def lowercase_and_stem(_words):
    """lowercase and stem sequence of words."""
    return [stemmer.stem(w.lower()) for w in _words]

def contains(subseq, inseq):
    return any(inseq[pos:pos + len(subseq)] == subseq for pos in range(0, len(inseq) - len(subseq) + 1))

def preprocess_phrases(phrases):
    pre_phrases = [' '.join(lowercase_and_stem(tokenize(phrase))) for phrase in phrases]
    #print("B", pre_phrases)
    # remove duplicate
    pre_phrases = list(dict.fromkeys(pre_phrases))
    # remove empty phrases
    pre_phrases = list(filter(None, pre_phrases))
    #print("A", pre_phrases)
    return pre_phrases
    #return [' '.join(lowercase_and_stem(phrase.split(" "))) for phrase in phrases]

def evaluate(top_N_keyphrases, references):
    if not len(top_N_keyphrases):
        return (0.0, 0.0, 0.0)
    P = len(set(top_N_keyphrases) & set(references)) / len(top_N_keyphrases)
    R = len(set(top_N_keyphrases) & set(references)) / len(references)
    F = (2*P*R)/(P+R) if (P+R) > 0 else 0
    return (P, R, F)

# load output file
top_m = {}
top_k = {}

with open(args.system, 'r') as f:
    for i,line in enumerate(f):
        if args.predictions_type != "txt":
            doc = json.loads(line)            
        else:
            doc = line
            #doc = re.sub("<peos>","",doc)
        top_m[doc["id"]] = preprocess_phrases(doc["top_m"])
        top_k[doc["id"]] = preprocess_phrases(doc["top_10"])
        #top_k[doc["id"]].extend([PAD_PHRASE for i in range(PAD_MIN-len(top_k[doc["id"]]))])

# load reference file
references = {}
tgt_pres_abs = defaultdict(list)
pre_pres_abs_top_m = defaultdict(list)
pre_pres_abs_top_k = defaultdict(list)

with open(args.reference, 'r') as f:
    for i,line in enumerate(tqdm(f)):
        doc = json.loads(line)
        
        # keywords / keyphrases switch
        if "keywords" in doc:
            keyphrases = doc["keywords"].split(";")
        else:
            keyphrases = doc["keyphrases"]

        # preprocess keyphrases
        references[doc["id"]] = preprocess_phrases(keyphrases)

        # preprocess title and abstract
        title = lowercase_and_stem(tokenize(doc["title"]))
        abstract = lowercase_and_stem(tokenize(doc["abstract"]))

        # check for present / absent keyphrases in references (tgt)
        for i,keyphrase in enumerate(references[doc["id"]]):
            tokens = keyphrase.split(" ")
            tgt_pres_abs[doc["id"]].append(0)
            ref_present_keyphrases = get_presents(doc)
            if i in ref_present_keyphrases:
                tgt_pres_abs[doc["id"]][i] = 1

        # check for present / absent keyphrases in top_m predicted (pre)
        for keyphrase in enumerate(top_m[doc["id"]]):
            #print(keyphrase)
            tokens = keyphrase[1].split(" ")
            pre_pres_abs_top_m[doc["id"]].append(0)
            if contains(tokens, title) or contains(tokens, abstract):
                pre_pres_abs_top_m[doc["id"]][-1] = 1

        # check for present / absent keyphrases in top_k predicted (pre)
        for keyphrase in top_k[doc["id"]]:
            tokens = keyphrase.split(" ")
            pre_pres_abs_top_k[doc["id"]].append(0)
            if contains(tokens, title) or contains(tokens, abstract):
                pre_pres_abs_top_k[doc["id"]][-1] = 1


# loop through the documents
scores_at_m = defaultdict(list)
scores_at_5 = defaultdict(list)
scores_at_10 = defaultdict(list)
valid_keys =  defaultdict(list)
generation_rates_at_m = defaultdict(list)
generation_rates_at_10 = defaultdict(list)
for i, docid in enumerate(tqdm(references)):

    # compute scores for all references
    scores_at_m['all'].append(evaluate(top_m[docid], references[docid]))
    scores_at_5['all'].append(evaluate(top_k[docid][:5], references[docid]))
    scores_at_10['all'].append(evaluate(top_k[docid][:10], references[docid]))
    valid_keys['all'].append(docid)

    # add scores for present and absent keyphrases
    pres_references = [phrase for j, phrase in enumerate(references[docid]) if tgt_pres_abs[docid][j]]
    pres_top_m = [phrase for j, phrase in enumerate(top_m[docid]) if pre_pres_abs_top_m[docid][j]]
    pres_top_k = [phrase for j, phrase in enumerate(top_k[docid]) if pre_pres_abs_top_k[docid][j]]
    pres_top_k.extend([PAD_PHRASE for j in range(PAD_MIN-len(pres_top_k))])
    if len(pres_references):
        scores_at_m['pre'].append(evaluate(pres_top_m, pres_references))
        scores_at_5['pre'].append(evaluate(pres_top_k[:5], pres_references))
        scores_at_10['pre'].append(evaluate(pres_top_k[:10], pres_references))
        valid_keys['pre'].append(docid)

    abs_references = [phrase for j, phrase in enumerate(references[docid]) if not tgt_pres_abs[docid][j]]
    abs_top_m = [phrase for j, phrase in enumerate(top_m[docid]) if not pre_pres_abs_top_m[docid][j]]
    abs_top_k = [phrase for j, phrase in enumerate(top_k[docid]) if not pre_pres_abs_top_k[docid][j]]
    generation_rates_at_m[docid] = len(abs_top_m) * 100/ len(top_m[docid])
    generation_rates_at_10[docid] = len(abs_top_k) * 100 / len(top_k[docid])

    abs_top_k.extend([PAD_PHRASE for j in range(PAD_MIN-len(pres_top_k))])
    if len(abs_references):
        scores_at_m['abs'].append(evaluate(abs_top_m, abs_references))
        scores_at_5['abs'].append(evaluate(abs_top_k[:5], abs_references))
        scores_at_10['abs'].append(evaluate(abs_top_k[:10], abs_references))
        valid_keys['abs'].append(docid)

# compute the average scores
for eval in ['all', 'pre', 'abs']:
    avg_scores_at_m = np.mean(scores_at_m[eval], axis=0)
    avg_scores_at_5 = np.mean(scores_at_5[eval], axis=0)
    avg_scores_at_10 = np.mean(scores_at_10[eval], axis=0)

            
    # print out the performance of the model
    print("{} F@M: {:>4.1f} F@5: {:>4.1f} F@10: {:>4.1f} - {}".format(eval, avg_scores_at_m[2]*100, avg_scores_at_5[2]*100, avg_scores_at_10[2]*100, args.system.split("/")[-1]))

    if args.output_scores != None:
        output_file = re.sub("top_[a-z-A-Z]+\.jsonl$", "", args.system) + "results.{}.csv".format(eval)
        print(output_file)
        with open(output_file, 'w') as f:
            for i, docid in enumerate(valid_keys[eval]):
                f.write("{}\t{}\t{}\t{}\n".format(docid, scores_at_m[eval][i][2], scores_at_5[eval][i][2], scores_at_10[eval][i][2]))

print("Generation rate @M: {}\n".format(np.mean(list(generation_rates_at_m.values()))))
print("Generation rate @10: {}\n".format(np.mean(list(generation_rates_at_10.values()))))

if args.output_scores != None:
    with open(re.sub("top_[a-z-A-Z]+\.jsonl$", "", args.system) + "gen_rate_@m.json","w") as o_m:
        json.dump(generation_rates_at_m,o_m)

    with open(re.sub("top_[a-z-A-Z]+\.jsonl$", "", args.system) + "gen_rate_@10.json","w") as o_ten:
        json.dump(generation_rates_at_10,o_ten)

