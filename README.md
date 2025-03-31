# Self-Compositional Data Augmentation for Scientific Keyphrase Generation

This is the repository for the article "Self-Compositional Data Augmentation for Scientific Keyphrase Generation" accepted at JCDL 2024.

## Datasets
Training and testing data
- KP20k: [https://huggingface.co/datasets/taln-ls2n/kp20k](https://huggingface.co/datasets/taln-ls2n/kp20k)
- KPTimes: [https://huggingface.co/datasets/taln-ls2n/kptimes](https://huggingface.co/datasets/taln-ls2n/kptimes)
- KPBiomed: [https://huggingface.co/datasets/taln-ls2n/kpbiomed](https://huggingface.co/datasets/taln-ls2n/kpbiomed)

Testing data:
- Nus: [https://huggingface.co/datasets/memray/krapivin](https://huggingface.co/datasets/memray/nus)
- Krapivin: [https://huggingface.co/datasets/memray/krapivin](https://huggingface.co/datasets/memray/krapivin)
- SemEval 2010: [https://huggingface.co/datasets/taln-ls2n/semeval-2010-pre](https://huggingface.co/datasets/taln-ls2n/semeval-2010-pre)

## Requirements
To install the required packages, run the following command:
`` pip install -r requirements.txt ``

## Running the scripts
To get the augmentation data, run the script named *preprocessing_augmentation.py*.
If you want to change the percentage of keyphrases in common for the file pairs, change *n* in line 52

example to run the script
``python src/preprocessing_augmentation.py --data_file data/kp20k/train.jsonl --output_file data/kp20k/augmentation_training_data.jsonl``

To run inference
```python inference.py```
This inference script is meant to do the inference of several models in a row, hence the loops.

To get the top k predictions
```python src/get_top_k.py --pred_file model_outputs/kp20k/bart_predictions.jsonl --reference_file data/testsets/kp20k/test.jsonl --file_type json --output_file model_outputs/kp20k/bart_top_preds.jsonl```
This will take your predictions (if you use copyrnn models that generate in a txt file, change to ``file_type txt``) and get the top k and put it in a json file for you evaluations.

To run evaluation
```python src/evaluation.py  --reference data/testsets/kp20k/test.jsonl --system model_outputs/kp20k/bart_top_preds.jsonl --output_scores yes```
If you put ``output_scores no`` the score will just be printed in the console but not saved in csv files.


##Cite this work
If this work is of interest to you, please use the following citation

```
@inbook{10.1145/3677389.3702504,
author = {Houbre, Ma\"{e}l and Boudin, Florian and Daille, B\'{e}atrice and Aizawa, Akiko},
title = {Self-Compositional Data Augmentation for Scientific Keyphrase Generation},
year = {2025},
isbn = {9798400710933},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3677389.3702504},
booktitle = {Proceedings of the 24th ACM/IEEE Joint Conference on Digital Libraries},
articleno = {7},
numpages = {10}
}
```