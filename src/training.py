from datasets import load_dataset, concatenate_datasets, dataset_dict
import sys
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, BartForConditionalGeneration
import argparse

def join_keyphrases(dataset):
    dataset["keyphrases"] = ";".join(dataset["keyphrases"])
    return dataset

# Getting the text from the title and the abstract
def get_text(dataset):
    dataset["text"] = dataset["title"] + "<s>" + dataset["abstract"]
    return dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
    description='prmu statistics',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-d","--dataset")
    parser.add_argument("-o","--output_dir")
    parser.add_argument("-a","--augmentation_file")

    args = parser.parse_args()

    train_dataset = load_dataset("json",data_files = {"train":"data/{}/train.jsonl".format(args.dataset), "val":"data/{}/val.jsonl".format(args.dataset)})
   
    train_dataset = train_dataset.map(get_text)
    train_dataset = train_dataset.map(join_keyphrases)


    if args.augmentation_file != None: # If we want to train a model with the augmented dataset

        augmentation_dataset = load_dataset("json", data_files = args.augmentation_file)

        #train_dataset = train_dataset.remove_columns(["id","title","abstract","prmu"])

        final_dataset = concatenate_datasets([train_dataset["train"],augmentation_dataset["train"]])
    
    else:
        final_dataset = train_dataset

    final_dataset = final_dataset.shuffle(seed=42)

    print(final_dataset)

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

    tokenized_datasets= final_dataset.map(preprocess_function, batched=True, num_proc = 10, desc="Running tokenizer on dataset")
    
    tokenized_datasets.set_format("torch")

    # Training arguments

    model = BartForConditionalGeneration.from_pretrained("../huggingface/bart-base")

    output_dir=args.output_dir

    tokenized_datasets = tokenized_datasets.remove_columns(
        final_dataset.column_names
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
            num_train_epochs=10,
            # Adjust batch size if this doesn't fit on the Colab GPU
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,  
            prediction_loss_only=True,
            load_best_model_at_end = False
        ),
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets["val"]
    )


    trainer.train()
    trainer.save_model(output_dir + "/final_model")
