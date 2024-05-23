from datasets import load_dataset
import spacy
import argparse
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

parser = argparse.ArgumentParser()

parser.add_argument("-d",
                    "--data_file")

parser.add_argument("-n",
                    "--num_proc")

args = parser.parse_args()

data = load_dataset("json",data_files=args.data_file)

data = data.map(lambda ex:{"title":ex["text"].split("<s>")[0]})
data = data.map(lambda ex:{"abstract":ex["text"].split("<s>")[1]})

data = data.map(lambda ex:{"tok_title": [token.text for token in nlp(ex["title"]) if not token.is_stop]},num_proc= args.num_proc)
data = data.map(lambda ex:{"tok_abstract": [token.text for token in nlp(ex["abstract"]) if not token.is_stop]},num_proc= args.num_proc)

data = data.map(lambda ex:{"intersection": len(set(ex["tok_title"]).intersection(set(ex["tok_abstract"])))/len(ex["tok_title"])})


# FILTERING 20 percent
print("FILTERING 20 PERCENT")
data_fil = data.filter(lambda ex:ex["intersection"] >= 0.20)

data_fil["train"].to_json("data/{}.jsonl".format(Path(args.data_file).stem))