from datasets import load_dataset, concatenate_datasets, dataset_dict
import sys
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, BartForConditionalGeneration, EarlyStoppingCallback

def join_keyphrases(dataset):
    dataset["keyphrases"] = ";".join(dataset["keyphrases"])
    return dataset

# Getting the text from the title and the abstract
def get_text(dataset):
    dataset["text"] = dataset["title"] + "<s>" + dataset["abstract"]
    return dataset

def prepare_input(dataset,sp_token):
    dataset["text"] = sp_token + dataset["text"]
    return dataset

gen_train_dataset = load_dataset("json",data_files="improving_from_relation/data/train_{}_generation.jsonl".format(sys.argv[1])) #,cache_dir="/gpfswork/rech/rgh/udr36oj/improving_from_relation")
relation_train_dataset = load_dataset("json", data_files = "improving_from_relation/data/train_{}_{}common_keyphrases.jsonl".format(sys.argv[1],sys.argv[2])) #,cache_dir="/gpfswork/rech/rgh/udr36oj/improving_from_relation")
val_gen_dataset = load_dataset("json",data_files="improving_from_relation/data/val_{}_generation.jsonl".format(sys.argv[1])) #,cache_dir="/gpfswork/rech/rgh/udr36oj/improving_from_relation")

gen_train_dataset = gen_train_dataset.remove_columns(["id","title","abstract","prmu"])
relation_train_dataset = relation_train_dataset.rename_column("label","keyphrases")

#gen_train_dataset = gen_train_dataset.map(prepare_input,fn_kwargs={"sp_token" : "<|KP|>"})
#relation_train_dataset = relation_train_dataset.map(prepare_input,fn_kwargs={"sp_token" : "<|COMMON|>"})
#val_gen_dataset = val_gen_dataset.map(prepare_input,fn_kwargs={"sp_token" : "<|KP|>"})

full_dataset = concatenate_datasets([gen_train_dataset["train"],relation_train_dataset["train"]])
full_dataset.shuffle(seed=42)

print(full_dataset)

# Making the references sequences
#dataset = dataset.map(join_keyphrases,num_proc=8,desc="Putting all keyphrases in a single sequence separated by ';' ")

# Loading the model
tokenizer = AutoTokenizer.from_pretrained("../huggingface/scibart-base")


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

tokenized_datasets= full_dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on dataset")
val_gen_dataset = val_gen_dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on dataset")

tokenized_datasets.set_format("torch")
val_gen_dataset.set_format("torch")

# Training arguments

model = BartForConditionalGeneration.from_pretrained("../huggingface/bart-base")

print("model loaded")

tokenized_datasets = tokenized_datasets.remove_columns(
    full_dataset.column_names
)

val_gen_dataset = val_gen_dataset.remove_columns(
    full_dataset.column_names
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="improving_from_relation/models/mtl_monodataset/bart-kp20k/3common",
        overwrite_output_dir=True,
        learning_rate=1e-4,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        weight_decay=0.01,
        max_steps=150000,
        eval_steps=15000,
        warmup_steps=150,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,  
        prediction_loss_only=True,
        save_steps=15000,
        load_best_model_at_end = False
    ),
    #data_collator=DefaultDataCollator(),#NLPDataCollator(),
    train_dataset=tokenized_datasets,
    eval_dataset=val_gen_dataset["train"],
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]    
)


trainer.train()
trainer.save_model("improving_from_relation/models/mtl_monodataset/bart-{}/{}common/final_model".format(sys.argv[1],sys.argv[2]))
