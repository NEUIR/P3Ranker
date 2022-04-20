from os import truncate
from typing import List, Tuple, Dict, Any

import json

import torch
from torch.utils.data import Dataset

from transformers import T5Tokenizer

class MNLIDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: T5Tokenizer,
        template,
        max_input: int = 1280000,
        original_t5:bool=False
    ) -> None:
        if not original_t5:
            self._label_mapping = ['true', 'neutral', 'false']
        else:
            self._mnli_label_mapping = ['true', 'neutral', 'false']
            self._nq_label_mapping=['false','true']
            self._anchor_label_mapping=['false','true']
        #self._label_mapping=['false','true']
        #对应[1176,7163,6136]
        #print(self._label_mapping)
        self._original_t5=original_t5
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_input = max_input
        self._template=template
        with open(self._dataset,'r') as f:
            self._examples=[eval(line) for line in f][:max_input]        

    # def __getitem__(self, index: int) -> Dict[str, Any]:
    #     example = self._examples[index]
    #     text='Premise: '+example["premise"]+' Hypothesis: '+example["hypothesis"]+' Entailment: '
    #     output=self._tokenizer(text,padding="max_length",truncation=True,max_length=384)
    #     output.update({'decoder_input_ids':[0],'label':example['label']})
    #     return output

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        #text='mnli hypothesis: ' + example["hypothesis"] + ' premise: ' + example["premise"]+' entailment: ' 
        if self._original_t5:
            if example['task']=='mnli':
                text='mnli hypothesis: ' + example["hypothesis"] + ' premise: ' + example["premise"]+" entailment: "
                label_mapping=self._mnli_label_mapping
            elif example['task']=='nq':
                label_mapping=self._nq_label_mapping
                text='NQ Question: '+example['query']+' Document: '+example['doc']+" answered: "
            elif example['task']=='anchor':
                label_mapping=self._anchor_label_mapping
                text='anchor: '+example['anchor']+' Document: '+example['doc']+" relevant: "
        else:
            text = self._template.replace("<h>", example["hypothesis"]).replace("<p>", example['premise'])
        #print(text)
        if example['task']=='mnli':
            hypothesis_tokenized=self._tokenizer(example['hypothesis'], padding="max_length", truncation=True, max_length=200)
            premise_tokenized=self._tokenizer(example['premise'],  padding="max_length", truncation=True, max_length=200)
            hypothesis_ids,hypothesis_attention_mask=hypothesis_tokenized['input_ids'][:-1],hypothesis_tokenized['attention_mask'][:-1]
            premise_ids,premise_attention_mask=premise_tokenized['input_ids'][:-1],premise_tokenized['attention_mask'][:-1]
        elif example['task']=='nq':
            hypothesis_tokenized=self._tokenizer(example['doc'], padding="max_length", truncation=True, max_length=200)
            premise_tokenized=self._tokenizer(example['query'],  padding="max_length", truncation=True, max_length=200)
            hypothesis_ids,hypothesis_attention_mask=hypothesis_tokenized['input_ids'][:-1],hypothesis_tokenized['attention_mask'][:-1]
            premise_ids,premise_attention_mask=premise_tokenized['input_ids'][:-1],premise_tokenized['attention_mask'][:-1]
        elif example['task']=='anchor':
            hypothesis_tokenized=self._tokenizer(example['doc'], padding="max_length", truncation=True, max_length=200)
            premise_tokenized=self._tokenizer(example['anchor'],  padding="max_length", truncation=True, max_length=200)
            hypothesis_ids,hypothesis_attention_mask=hypothesis_tokenized['input_ids'][:-1],hypothesis_tokenized['attention_mask'][:-1]
            premise_ids,premise_attention_mask=premise_tokenized['input_ids'][:-1],premise_tokenized['attention_mask'][:-1]
        #hypothesis_ids=self._tokenizer(example['hypothesis'],padding="max_length", truncation=True, max_length=50)['input_ids'][:-1]
        #self._tokenizer(example['hypothesis'],padding="max_length", truncation=True, max_length=50)['attention_mask'][:-1]
        #premise_ids=self._tokenizer(example['premise'],padding="max_length", truncation=True, max_length=320)['input_ids'][:-1]
        #text='hypothesis: ' + example["hypothesis"] + ' premiseument: ' + example["premise"]+" Relevant: "
        tokenized = self._tokenizer(text, padding="max_length", truncation=True, max_length=512)
        source_ids, source_mask = tokenized["input_ids"], tokenized["attention_mask"]
        tokenized = self._tokenizer(label_mapping[example["label"]], padding="max_length", truncation=True, max_length=10)
        target_ids = tokenized["input_ids"]
        target_ids = [
           (label if label != self._tokenizer.pad_token_id else -100) for label in target_ids
        ]
        raw_label = label_mapping[example["label"]]
        output = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
            "raw_label": raw_label,
            'label_id': example['label'],
            'hypothesis_ids':hypothesis_ids,
            'premise_ids':premise_ids,
            'hypothesis_attention_mask':hypothesis_attention_mask,
            'premise_attention_mask':premise_attention_mask,
        }
        return output

    def __len__(self) -> int:
        return len(self._examples)


    def collate(self, batch: Dict[str, Any]):
        input_ids = torch.tensor([item['input_ids'] for item in batch])
        hypothesis_ids=torch.tensor([item['hypothesis_ids'] for item in batch])
        premise_ids=torch.tensor([item['premise_ids'] for item in batch])
        attention_mask = torch.tensor([item['attention_mask'] for item in batch])
        hypothesis_attention_mask = torch.tensor([item['hypothesis_attention_mask'] for item in batch])
        premise_attention_mask = torch.tensor([item['premise_attention_mask'] for item in batch])
        labels = torch.tensor([item['labels'] for item in batch])
        raw_label = [item["raw_label"] for item in batch]
        label_id=torch.tensor([item['label_id'] for item in batch])
        return {'input_ids': input_ids,"hypothesis_ids":hypothesis_ids ,"premise_ids":premise_ids,"attention_mask": attention_mask, 'labels': labels, 
        "raw_label": raw_label,"label_id":label_id,
        "hypothesis_attention_mask":hypothesis_attention_mask,"premise_attention_mask":premise_attention_mask
        }