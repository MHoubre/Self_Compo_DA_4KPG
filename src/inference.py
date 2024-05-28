#!/usr/bin/env python
# coding: utf-8

import json
from transformers import BartConfig
from transformers import BartModel, BartForConditionalGeneration
from transformers import BartTokenizerFast, AutoTokenizer
import datasets
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, ReadInstruction
from tokenizers import Tokenizer
from datasets import load_metric
from glob import glob
#from data_processing_utils import get_input

from pathlib import Path
import torch
import numpy as np
import random
import re


def prepare_input(dataset,sp_token="<s>"):
    dataset["text"] = dataset["title"]+ sp_token + dataset["abstract"]
    return dataset

def generate_keyphrases(batch, key):
    
    
    inputs = tokenizer(
        batch[key],padding="max_length",max_length= 512,truncation=True, return_tensors='pt'
    )
    
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

                             
    outputs = model.generate(inputs=input_ids,attention_mask=attention_mask,
                             num_beams=20,
                             num_return_sequences=20
    )
    # all special tokens including will be removed
    output_strs = tokenizer.batch_decode(outputs,skip_special_tokens=True)

    batch["pred"] = output_strs

    return batch

if __name__ == "__main__":

    for training_type in ["3common"]:

        for kpdata in ["kp20k"] : #,"kpbiomed_small"]:

            print(kpdata)
            print(training_type)
            model_list = glob("models/filter_training/bart-{}/50ksteps/{}/final*".format(kpdata,training_type))
            print(model_list)
            model_list = [Path(element).stem for element in model_list]
            print(model_list)

            for checkpoint in model_list:

                model_path = "models/filter_training/bart-{}/50ksteps/{}/{}".format(kpdata,training_type,checkpoint)

                dataset = load_dataset("json",data_files={"test":"data/test_{}.jsonl".format(kpdata)})

                print("MODEL: {}".format(checkpoint))

                dataset = dataset.map(prepare_input,fn_kwargs={"sp_token":"<s>"})
                print("PROCESSING DONE\n")

                tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

                model = BartForConditionalGeneration.from_pretrained(model_path)

                model.to("cuda")

                dataset = dataset.map(prepare_input,num_proc=6)

                dataset = dataset.map(generate_keyphrases,fn_kwargs={"key":"text"})

                dataset["train"].to_json("generated/filtering/{}/{}/output_final.jsonl".format(kpdata,training_type))