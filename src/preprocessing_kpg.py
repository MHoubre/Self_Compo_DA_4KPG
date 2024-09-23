import os
import shutil
import sys
from datasets import load_dataset
from data_processing_utils import meng17_tokenize_column
from argparse import ArgumentParser
from pathlib import Path

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


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("-d","--data_file",help="JsonL data file to be processed")
    parser.add_argument("-o","--output_file",help="Processed JsonL file")

    args = parser.parse_args()

    data = load_dataset("json", data_files=args.data_file)
    data = data.map(get_input, num_proc=8) # We get the input text at the right format (title + abstract)
    data = data.map(join_keyphrases,num_proc=8) # We put the list of keyphrases in a single sequence

    filename = Path(args.data_file).stem()
    
    data["train"].to_json(args.output_file)
