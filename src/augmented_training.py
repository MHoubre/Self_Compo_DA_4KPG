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

gen_train_dataset = load_dataset("json",data_files="data/train_{}_generation.jsonl".format(sys.argv[1]))
relation_train_dataset = load_dataset("json", data_files = "data/train_{}_{}.jsonl".format(sys.argv[1],sys.argv[2]))
val_gen_dataset = load_dataset("json",data_files="data/val_{}_generation.jsonl".format(sys.argv[1]))

#gen_train_dataset = gen_train_dataset.remove_columns(["id","title","abstract","prmu"])
relation_train_dataset = relation_train_dataset.rename_column("label","keyphrases")

relation_train_dataset = relation_train_dataset.map(join_keyphrases,num_proc=8,desc="Putting all keyphrases in a single sequence separated by ';' ")

full_dataset = concatenate_datasets([gen_train_dataset["train"],relation_train_dataset["train"]])
full_dataset = full_dataset.shuffle(seed=42)

print(full_dataset)

# Loading the model
tokenizer = AutoTokenizer.from_pretrained("../huggingface/bart-base")


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

tokenized_datasets= full_dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on train dataset")
val_gen_dataset = val_gen_dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on val dataset")

tokenized_datasets.set_format("torch")
val_gen_dataset.set_format("torch")

# Training arguments

model = BartForConditionalGeneration.from_pretrained("../huggingface/bart-base")

output_dir="models/augmented/bart-{}/steps/{}/".format(sys.argv[1],sys.argv[2])

tokenized_datasets = tokenized_datasets.remove_columns(
    full_dataset.column_names
)

val_gen_dataset = val_gen_dataset.remove_columns(
    full_dataset.column_names
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=1e-4,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        weight_decay=0.01,
        #num_train_epochs=10,
        num_train_epochs=10,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,  
        prediction_loss_only=True,
        save_steps=150000,
        load_best_model_at_end = False
    ),
    train_dataset=tokenized_datasets,
    eval_dataset=val_gen_dataset["train"]
)


trainer.train()
trainer.save_model(output_dir + "/final_model")
