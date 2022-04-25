# Prefinetune 

# Implementation for Prefinetune. 

## Dataset

See https://huggingface.co/datasets/multi_nli to know more about the dataset.

We use the train split as training set and validation_mismatched split as dev set.


## Models

[mnli_model.py](mnli_model.py) describes how our models designed. 

## Inputs

[mnli_dataset.py](mnli_dataset.py) describes how the original dataset is organized to input to the models.

## Run

```bash
bash train_mnli.sh 
```

## Expansion

This code supports for every text dataset. Go to [model](mnli_model.py) and [dataset][mnli_dataset.py] to see details.
