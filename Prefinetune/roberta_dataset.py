from typing import List, Tuple, Dict, Any

import json


import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

class RobertaDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        tokenizer: AutoTokenizer,
        max_input: int = 1280000,
        task: str = 'ranking',
        template: str = None,
    ) -> None:
        self._dataset = dataset
        self._tokenizer = tokenizer
        self._max_input = max_input
        self._task = task
        # if self._seq_max_len > 512:
        #     raise ValueError('premise_max_len + hypothesis_max_len + 4 > 512.')
        assert task is not 'ranking'
        if self._task.startswith("prompt"):
            assert template is not None
            self._template = template
        with open(self._dataset, 'r') as f:
            self._examples = []
            for i, line in enumerate(f):
                if i >= self._max_input:
                    break
                line = json.loads(line)
                self._examples.append(line)
        self._count = len(self._examples)

    def collate(self, batch: Dict[str, Any]):
        if self._task == 'classification':
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            label = torch.tensor([item['label_id'] for item in batch])
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label_id': label}
        elif self._task == "prompt_classification":

            input_ids = torch.tensor([item['input_ids'] for item in batch])
            premise_ids = torch.tensor([item['premise_ids'] for item in batch])
            hypothesis_ids = torch.tensor([item['hypothesis_ids'] for item in batch])

            mask_pos = torch.tensor([item['mask_pos'] for item in batch])

            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            premise_attention_mask = torch.tensor([item['premise_attention_mask'] for item in batch])
            hypothesis_attention_mask = torch.tensor([item['hypothesis_attention_mask'] for item in batch])


            label = torch.tensor([item['label_id'] for item in batch])

            return {
            'input_ids': input_ids, 'premise_ids': premise_ids,'hypothesis_ids': hypothesis_ids,
            'attention_mask': attention_mask,'premise_attention_mask': premise_attention_mask,'hypothesis_attention_mask': hypothesis_attention_mask,
             "mask_pos": mask_pos, 'label_id': label
             }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._examples[index]
        if self._task == 'classification':
            tokenizer_output = self._tokenizer(example["premise"], example["hypothesis"], padding="max_length", truncation="only_second", max_length=512)
            output = {"label_id": example["label"]}
            output.update(tokenizer_output)
            return output
        elif self._task == "prompt_classification":
            hypothesis = example["hypothesis"].strip()
            premise=example['premise']
            text = self._template.replace("<p>", premise).replace("<h>", hypothesis).replace("[MASK]", "<mask>")

            text_tokenized = self._tokenizer(text, padding="max_length", truncation=True, max_length=512)
            premise_tokenized=self._tokenizer(premise, truncation=True, max_length=200)
            hypothesis_tokenized=self._tokenizer(hypothesis, padding="max_length",truncation=True, max_length=200)

            input_ids = text_tokenized.input_ids
            premise_ids = premise_tokenized.input_ids[:-1]
            premise_ids=premise_ids+(200-len(premise_ids))*[self._tokenizer.pad_token_id]
            hypothesis_ids = hypothesis_tokenized.input_ids[1:]

            attention_mask = text_tokenized.attention_mask
            premise_attention_mask=premise_tokenized.attention_mask[:-1]
            premise_attention_mask=premise_attention_mask+(200-len(premise_attention_mask))*[0]
            hypothesis_attention_mask=hypothesis_tokenized.attention_mask[1:]

            mask_pos = input_ids.index(self._tokenizer.mask_token_id)       #soft mask_pos should be re-calculated in model.py
            output = {
                "input_ids": input_ids, "premise_ids": premise_ids, "hypothesis_ids": hypothesis_ids,  
                "attention_mask": attention_mask, "premise_attention_mask": premise_attention_mask, "hypothesis_attention_mask": hypothesis_attention_mask, 
            "label_id": example["label"], "mask_pos": mask_pos
            }
            # output.update(tokenizer_output)
            return output



    def __len__(self) -> int:
        return self._count
