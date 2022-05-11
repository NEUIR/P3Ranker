# P3 Ranker

P<sup>3</sup> Ranker is a few-shot neural-ir ranker based on pretrained sentence-to-sentence transformers:T5. It's superior performance on few-shot scenatios benefit from the P<sup>3</sup> training paradigm: Pretraining - Prefinetuning- Prompt-based finetuning.

See my publication to get more information [*P3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning*](https://arxiv.org/pdf/2205.01886.pdf)
### Project Structures
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
│   ├── metrics
│   │    ├── __init__.py
│   │    └── metric.py
│   ├── models
│   │    ├── __init__.py
│   │    ├── bert_maxp.py
│   │    ├── bert_prompt_.py
│   │    ├── bert.py
│   │    └── t5.py
│   ├── __init__.py
│   └── utils.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py 
```

### Prerequisites
Install dependencies:

```bash
git clone https://github.com/NEUIR/P3Ranker.git
cd P3-Rankers
pip install -r requirements.txt
```


### Data Preparation
Each data sample should be in the following format and be collected in a 'jsonl' file. 
```bash
{"query": "elegxo meaning", "doc": "Swedish Meaning: The name Sonia is a Swedish baby name. In Swedish the meaning of the name Sonia is: Wise. American Meaning: The name Sonia is an American baby name.In American the meaning of the name Sonia is: Wise. Russian Meaning: The name Sonia is a Russian baby name. In Russian the meaning of the name Sonia is: Wisdom.Greek Meaning: The name Sonia is a Greek baby name. In Greek the meaning of the name Sonia is: Wisdom; wise.he name Sonia is a Swedish baby name. In Swedish the meaning of the name Sonia is: Wise. American Meaning: The name Sonia is an American baby name.", "label": 0, "query_id": 1183785, "doc_id": 2560705}
```
We will release our few-shot dataset soon.

### Prompt Generation

Details about the Discrete Prompt Generation can be find in https://github.com/princeton-nlp/LM-BFF and our paper

### Prefinetune 

```bash
cd Reproduce
```
And you will find how to do prefinetune.
### Reproduce our results

Directly run the scripts we stored in './commands' can reproduce our results. One example is shown below:

```bash
bash commands/bert.sh 5
```
The above command is for reproducing results in our 5-q few-shot scenarios mentioned in our paper. 

### Citation
If you use our code our our data for your research, feel free to cite our publication: 
```bash
@article{hu2022p,
  title={P $\^{} 3$ Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning},
  author={Hu, Xiaomeng and Yu, Shi and Xiong, Chenyan and Liu, Zhenghao and Liu, Zhiyuan and Yu, Ge},
  journal={arXiv preprint arXiv:2205.01886},
  year={2022}
}
```
### Contact 

Please send email to hxm183083@gmail.com.

