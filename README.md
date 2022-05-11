# P3 Ranker

P<sup>3</sup> Ranker is a few-shot neural-ir ranker based on pretrained sentence-to-sentence transformers: T5. It's superior performances on few-shot scenarios benefit from the P<sup>3</sup> training paradigm: **P**retraining -> **P**refinetuning -> **P**rompt-based finetuning.

![image](https://github.com/NEUIR/P3Ranker/blob/main/msmarco_results.png)

See my publication to get more information

* [*P<sup>3</sup> Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning*](https://arxiv.org/pdf/2205.01886.pdf)
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
cd P3Ranker
pip install -r requirements.txt
```


### Data Preparation
Each data sample should be in the following format and be collected in a 'jsonl' file. 
```bash
{"query": "elegxo meaning", "doc": "Swedish Meaning: The name Sonia is a Swedish baby name. In Swedish the meaning of the name Sonia is: Wise. American Meaning: The name Sonia is an American baby name.In American the meaning of the name Sonia is: Wise. Russian Meaning: The name Sonia is a Russian baby name. In Russian the meaning of the name Sonia is: Wisdom.Greek Meaning: The name Sonia is a Greek baby name. In Greek the meaning of the name Sonia is: Wisdom; wise.he name Sonia is a Swedish baby name. In Swedish the meaning of the name Sonia is: Wise. American Meaning: The name Sonia is an American baby name.", "label": 0, "query_id": 1183785, "doc_id": 2560705}
```
We will release our few-shot dataset soon.

### Prompt Selection
* Manual Prompt
  1. For encoder-only-models (BERT and RoBERTa) we use :
  ``` bash 
  [q] is [MASK] (relevant|irrelevant) to [d]  
  ```
  2. For encoder-decoder models (T5 and P<sup>3</sup> Ranker) we use:
  ```bash
  Query: [q] Document: [d] Relevant
  ```
 * Auto Prompt
Details about the Discrete Prompt Generation can be find in https://github.com/princeton-nlp/LM-BFF and our paper


### Prefinetune 

```bash
cd Reproduce
```
And you will find how to do prefinetune.
### Reproduce our results

Directly run the scripts we stored in [commands](https://github.com/NEUIR/P3Ranker/tree/main/commands) can reproduce our results. One example is shown below:

```bash
set -ex
export OMP_NUM_THREADS=1
LR=2e-5
EPOCH=1000000                           #set EPOCH to a large number so that the training process can only be limited by the MAX_STEPS
Q=$1
if [ $Q == 'full' ];then
    MAX_STEPS=50000
    devq=500
    LOG_STEP=500
    EVAL_EVERY=5000
    BATCH_SIZE=8
    metric='mrr_cut_10'
fi
if [ $Q == 1000 ];then
    MAX_STEPS=3000
    devq=500
    LOG_STEP=100
    EVAL_EVERY=300
    BATCH_SIZE=8
    metric='mrr_cut_10'
fi 
if [ $Q == 50 ];then
    MAX_STEPS=3000
    devq=50
    LOG_STEP=100
    EVAL_EVERY=300
    BATCH_SIZE=2
    metric='mrr_cut_10'
fi
if [ $Q == 5 ];then
    MAX_STEPS=3000
    devq=5
    LOG_STEP=100
    EVAL_EVERY=300
    BATCH_SIZE=2
    metric='mrr_cut_1000'
fi
NEG=1
model="bert-base-ft"
ckpt="/data/home/scv0540/run/pretrained_models/bert-base"
dir_path="/data/home/scv0540/run/promptir"
python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=3119 \
train.py \
-task classification \
-model bert \
-qrels $dir_path/collections/msmarco-passage/qrels.train.tsv     \
-train $dir_path/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
-dev $dir_path/dataset/msmarco/dev/$devq-q.jsonl  \
-test $dir_path/dataset/msmarco/test/all-q.jsonl  \
-max_input 80000000 \
-vocab $ckpt  \
-pretrain $ckpt  \
-metric $metric  \
-max_query_len 50  \
-max_doc_len 400 \
-epoch $EPOCH  \
-batch_size $BATCH_SIZE  \
-lr $LR  \
-eval_every $EVAL_EVERY  \
-optimizer adamw   \
-dev_eval_batch_size 200  \
-n_warmup_steps 0  \
-logging_step $LOG_STEP  \
-save $dir_path/checkpoints/$model/q$Q-n-$NEG/  \
-res $dir_path/results/$model/q$Q-n-$NEG.trec  \
-test_res $dir_path/results/$model/test_q$Q-n-$NEG.trec  \
--log_dir=$dir_path/logs/$model/q$Q-n-$NEG/  \
--max_steps=$MAX_STEPS  \
--pos_word="relevant"  \
--neg_word="irrelevant"  \
--template='<q> is [MASK] (relevant|irrelevant) to <d>'  \
-gradient_accumulation_steps 1  \
--prefix='[1996, 3025, 6251, 2003, 1037, 23032, 1012, 23032, 2003]'   \
--infix='[11167,10]'    \
--suffix='[2000, 9986, 5657, 2213, 3372, 1012, 1996, 2279, 6251, 2003, 1037, 6254, 1012]'  \
--soft_sentence=""  \
```
The above command is for reproducing results in our 5-q few-shot scenarios mentioned in our paper. 

### Trained Checkpoints

We will release our trained checkpoints soon.

### Citation
If you use our code or our data for your research, feel free to cite our publication: 
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

