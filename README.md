# P3 Ranker
Implementation for our SIGIR2022 accepted paper:  

*P3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning*

## Project Structures
```bash
├── commands
│   ├── bert.sh
│   ├── p3ranker.sh
│   ├── prop_ft.sh
│   ├── roberta.sh
│   └── t5v11.sh
├── Prefinetune
│   ├── mnli_dataloader.py
│   ├── mnli_dataset.py
│   ├── mnli_model.py
│   ├── README.md
│   ├── train_mnli.sh
│   ├── train_nq.sh
│   ├── train.py
│   └── utils.py
├── src
│   ├── data
│   │    ├── datasets
│   │    │   ├── __init__.py
│   │    │   ├── bert_dataset.py
│   │    │   ├── bertmaxp_dataset.py
│   │    │   ├── dataset.py
│   │    │   ├── roberta_dataset.py
│   │    │   └── t5_dataset.py
│   │    └── tokenizers
│   │        ├── __init__.py
│   │        ├── tokenizer.py
│   │        └── word_tokenizer.py
│   ├── extractors
│   │    ├── __init__.py
│   │    └── classic_extractor.py
│   ├── metrics
│   │    ├── __init__.py
│   │    └── metric.py
│   ├── models
│   │    ├── __init__.py
│   │    ├── bert_maxp.py
│   │    ├── bert_prompt_.py
│   │    ├── bert.py
│   │    └── t5.py
│   ├── modules
│   │    ├── attentons
│   │    │   ├── __init__.py
│   │    │   ├── multi_head_attention.py
│   │    │   └── scaled_dot_product_attention.py
│   │    ├── embedders
│   │    │   ├── __init__.py
│   │    │   └── embedder.py
│   │    ├── encoders
│   │    │   ├── __init__.py
│   │    │   ├── cnn_encoder.py
│   │    │   ├── feed_forward_encoder.py
│   │    │   ├── positional_encoder.py
│   │    │   └── transformer_encoder.py
│   │    └── matchers
│   │        ├── __init__.py
│   │        └── kernel_matcher.py
│   ├── __init__.py
│   └── utils.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py 
```

## Prerequisites
Install dependencies:

```bash
git clone https://github.com/NEUIR/P3Ranker.git
cd P3-Rankers
pip install -r requirements.txt
```


## Data Preparation
We will release our few-shot dataset soon.

## Prompt Generation

Details about the Discrete Prompt Generation can be find in https://github.com/princeton-nlp/LM-BFF and our paper

## Prefinetune 

```bash
cd Reproduce
```
And you will find how to do prefinetune.
## Reproduce our results

Directly run the scripts we stored in './commands' can reproduce our results. One example is shown below:

```bash
bash commands/bert.sh 5
```
The above command is for reproducing results in our 5-q few-shot scenarios mentioned in our paper. 

## Contact 

Please send email to hxm183083@gmail.com.

