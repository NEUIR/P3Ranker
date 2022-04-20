from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle

from transformers import BertConfig, AutoModel, AutoModelForMaskedLM, AutoConfig,AutoTokenizer

class BertPrompt(nn.Module):
    def __init__(
        self,
        pretrained: str,
        soft_prompt: bool = False,
        prefix:str=None,
        suffix:str=None
    ) -> None:
        super(BertPrompt, self).__init__()
        self._pretrained = pretrained
        self._label_mapping=["Yes","Maybe","No"]
        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModelForMaskedLM.from_pretrained(self._pretrained, config=self._config)
        self._tokenizer=AutoTokenizer.from_pretrained(self._pretrained)
        self._soft_prompt = soft_prompt
        self.soft_embedding = None

        if self._soft_prompt:
            self.prefix_soft_index,self.suffix_soft_index=eval(prefix),eval(suffix)
            #[3,27569,10],[11167,10],[31484,17,10,1]
            self.p_num,self.s_num=len(self.prefix_soft_index),len(self.suffix_soft_index)

            self.prefix_soft_embedding_layer=nn.Embedding(
                self.p_num,self._config.hidden_size
                )
            self.suffix_soft_embedding_layer=nn.Embedding(
                self.s_num,self._config.hidden_size
                )
            self.normal_embedding_layer=self._model.get_input_embeddings()
            self.prefix_soft_embedding_layer.weight.data=torch.stack(
                [self.normal_embedding_layer.weight.data[i,:].clone().detach().requires_grad_(True) for i in self.prefix_soft_index]
                )
            self.suffix_soft_embedding_layer.weight.data=torch.stack(
                [self.normal_embedding_layer.weight.data[i,:].clone().detach().requires_grad_(True) for i in self.suffix_soft_index]
                )

            self.prefix_soft_ids=torch.tensor(range(self.p_num))
            self.suffix_soft_ids=torch.tensor(range(self.s_num))
            self.mask_ids=torch.tensor([self._tokenizer.mask_token_id])
            for param in self._model.parameters():
                param.requires_grad_(False)
    

    def forward(
        self, input_ids: torch.Tensor, premise_ids: torch.Tensor, hypothesis_ids: torch.Tensor, 
        masked_token_pos: torch.Tensor, 
        attention_mask: torch.Tensor = None, premise_attention_mask: torch.Tensor = None, hypothesis_attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size=input_ids.shape[0]
        if self._soft_prompt:
            prefix_soft_ids=torch.stack([self.prefix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            mask_ids=torch.stack([self.mask_ids for i in range(batch_size)]).to(input_ids.device)
            suffix_soft_ids=torch.stack([self.suffix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            
            prefix_soft_embeddings=self.prefix_soft_embedding_layer(prefix_soft_ids)
            suffix_soft_embeddings=self.suffix_soft_embedding_layer(suffix_soft_ids)
            
            premise_embeddings=self.normal_embedding_layer(premise_ids)
            hypothesis_embeddings=self.normal_embedding_layer(hypothesis_ids)
            mask_embeddings=self.normal_embedding_layer(mask_ids)
            
            input_embeddings=torch.cat(
                [premise_embeddings,prefix_soft_embeddings,mask_embeddings,suffix_soft_embeddings,hypothesis_embeddings],
                dim=1
                )
            
            prefix_soft_attention_mask=torch.ones(batch_size,self.p_num).to(input_ids.device)
            mask_attention_mask=torch.ones(batch_size,1).to(input_ids.device)
            suffix_soft_attention_mask=torch.ones(batch_size,self.s_num).to(input_ids.device)
            
            attention_mask=torch.cat(
                [premise_attention_mask,prefix_soft_attention_mask,mask_attention_mask,suffix_soft_attention_mask,hypothesis_attention_mask],
                dim=1
                )
            output = self._model(
                inputs_embeds=input_embeddings,attention_mask = attention_mask
                )[0]
            masked_token_pos=torch.full(masked_token_pos.shape,50+self.p_num).to(input_ids.device)
        else:
            output = self._model(input_ids, attention_mask = attention_mask)[0]
        vocab_size = output.shape[2]
        masked_token_pos = torch.unsqueeze(masked_token_pos, 1)
        masked_token_pos = torch.unsqueeze(masked_token_pos, 2)
        masked_token_pos = torch.stack([masked_token_pos] * vocab_size, 2)
        masked_token_pos = torch.squeeze(masked_token_pos, 3)
        masked_token_logits = torch.gather(output, 1, masked_token_pos)
        masked_token_logits=masked_token_logits.reshape(-1,vocab_size)
        logits = masked_token_logits[:, [self._tokenizer.encode(item)[1] for item in self._label_mapping]]
        return logits, masked_token_logits