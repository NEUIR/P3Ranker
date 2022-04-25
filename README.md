# P3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning
Implementation for our SIGIR2022 accepted paper:  

*P3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning*


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

