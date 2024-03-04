#!/usr/bin/env python
# coding: utf-8

from gc import callbacks
import json
import sys
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import Trainer, EarlyStoppingCallback
from transformers import BartModel, BartForConditionalGeneration
from transformers import BartTokenizer, AutoTokenizer
from transformers import TrainingArguments
from transformers.data.data_collator import InputDataClass, DefaultDataCollator
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, ReadInstruction
from datasets import load_metric

# Loading training dataset.
dataset = load_dataset("json", data_files={"train":"MTL_KRP/data/train_generation.jsonl", "validation" : "MTL_KRP/data/val_generation.jsonl"},cache_dir="/gpfswork/rech/rgh/udr36oj")

def join_keyphrases(dataset):
    dataset["keyphrases"] = ";".join(dataset["keyphrases"])
    return dataset

# Getting the text from the title and the abstract
def get_text(dataset):
    dataset["text"] = dataset["title"] + "<s>" + dataset["abstract"]
    return dataset


# Making the references sequences
#dataset = dataset.map(join_keyphrases,num_proc=8,desc="Putting all keyphrases in a single sequence separated by ';' ")

# Loading the model
tokenizer = AutoTokenizer.from_pretrained("huggingface/keybart")


# Function to tokenize the text using Huggingface tokenizer
def preprocess_function(dataset):

    model_inputs = tokenizer(
        dataset["text"],max_length= 512,padding="max_length",truncation=True
    )
    
    with tokenizer.as_target_tokenizer():
    
        labels = tokenizer(
            dataset["keyphrases"], max_length= 128, padding="max_length", truncation=True)
        

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    


#dataset = dataset.map(get_text,num_proc=10, desc="Getting full text (title+abstract)")

tokenized_datasets= dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on dataset")

tokenized_datasets.set_format("torch")

# Training arguments

model = BartForConditionalGeneration.from_pretrained("huggingface/keybart")


tokenized_datasets = tokenized_datasets.remove_columns(
    dataset["train"].column_names
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="MTL_KRP/models/keybart/singletask_model",
        overwrite_output_dir=True,
        learning_rate=1e-4,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        weight_decay=0.01,
        max_steps=300000,
        eval_steps=10000,
        warmup_steps=5000,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,  
        prediction_loss_only=True,
        save_steps=10000,
        load_best_model_at_end = True
    ),
    #data_collator=DefaultDataCollator(),#NLPDataCollator(),
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]    
)


trainer.train()
trainer.save_model("MTL_KRP/models/keybart/singletask_model/final_bart_model")
