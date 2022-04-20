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
        mode: str = 'cls',
        task: str = 'ranking',
        pos_word_id: int = 0,
        neg_word_id: int = 0,
        soft_prompt: bool = False,
        prefix:str=None,
        suffix:str=None
    ) -> None:
        super(BertPrompt, self).__init__()
        self._pretrained = pretrained
        self._mode = mode
        self._task = task

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModelForMaskedLM.from_pretrained(self._pretrained, config=self._config)
        self._tokenizer=AutoTokenizer.from_pretrained(self._pretrained)
        self._pos_word_id = pos_word_id
        self._neg_word_id = neg_word_id
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
            #print("soft prompt")
            #torch.manual_seed(13)
            #self.soft_embedding = nn.Embedding(100, self._config.hidden_size)
            #print(self.soft_embedding.weight.data)
            #self._model_embedding = self._model.get_input_embeddings()
            #self.soft_embedding.weight.data = self._model_embedding.weight.data[:100, :].clone().detach().requires_grad_(True)
            #print(self.soft_embedding.weight.data)
            #for param in self._model.parameters():
            #    param.requires_grad_(False)
            #torch.manual_seed(13)
            #self.new_lstm_head = nn.LSTM(
            #    input_size = self._config.hidden_size,
            #    hidden_size = self._config.hidden_size, # TODO P-tuning different in LAMA & FewGLUE
            #    # TODO dropout different in LAMA and FewGLUE
            #    num_layers=2,
            #    bidirectional=True,
            #    batch_first=True
            #)
            #print(self.new_lstm_head.all_weights[0][0][0])
            #torch.manual_seed(13)
            #self.new_mlp_head = nn.Sequential(
            #    nn.Linear(2*self._config.hidden_size, self._config.hidden_size),
            #    nn.ReLU(),
            #    nn.Linear(self._config.hidden_size, self._config.hidden_size)
            #)

    def forward(
        self, input_ids: torch.Tensor, query_ids: torch.Tensor, doc_ids: torch.Tensor, 
        masked_token_pos: torch.Tensor, 
        input_mask: torch.Tensor = None, query_input_mask: torch.Tensor = None, doc_input_mask: torch.Tensor = None, 
        segment_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size=input_ids.shape[0]
        if self._soft_prompt:
            prefix_soft_ids=torch.stack([self.prefix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            mask_ids=torch.stack([self.mask_ids for i in range(batch_size)]).to(input_ids.device)
            suffix_soft_ids=torch.stack([self.suffix_soft_ids for i in range(batch_size)]).to(input_ids.device)
            
            prefix_soft_embeddings=self.prefix_soft_embedding_layer(prefix_soft_ids)
            suffix_soft_embeddings=self.suffix_soft_embedding_layer(suffix_soft_ids)
            
            query_embeddings=self.normal_embedding_layer(query_ids)
            doc_embeddings=self.normal_embedding_layer(doc_ids)
            mask_embeddings=self.normal_embedding_layer(mask_ids)
            
            input_embeddings=torch.cat(
                [query_embeddings,prefix_soft_embeddings,mask_embeddings,suffix_soft_embeddings,doc_embeddings],
                dim=1
                )
            
            prefix_soft_attention_mask=torch.ones(batch_size,self.p_num).to(input_ids.device)
            mask_attention_mask=torch.ones(batch_size,1).to(input_ids.device)
            suffix_soft_attention_mask=torch.ones(batch_size,self.s_num).to(input_ids.device)
            
            attention_mask=torch.cat(
                [query_input_mask,prefix_soft_attention_mask,mask_attention_mask,suffix_soft_attention_mask,doc_input_mask],
                dim=1
                )
            output = self._model(
                inputs_embeds=input_embeddings,attention_mask = attention_mask
                )[0]
            masked_token_pos=torch.full(masked_token_pos.shape,50+self.p_num).to(input_ids.device)
        else:
            output = self._model(input_ids, attention_mask = input_mask)[0]
        if self._mode == 'cls':

            pass
        elif self._mode == 'pooling':

            pass
        else:
            raise ValueError('Mode must be `cls` or `pooling`.')
        
        

        vocab_size = output.shape[2]
        masked_token_pos = torch.unsqueeze(masked_token_pos, 1)
        masked_token_pos = torch.unsqueeze(masked_token_pos, 2)
        masked_token_pos = torch.stack([masked_token_pos] * vocab_size, 2)
        masked_token_pos = torch.squeeze(masked_token_pos, 3)
        masked_token_logits = torch.gather(output, 1, masked_token_pos)
        
        masked_token_logits=masked_token_logits.reshape(-1,vocab_size)
        logits = masked_token_logits[:, [self._neg_word_id, self._pos_word_id]]
        

        return logits, masked_token_logits

    def save_prompts(self, file):
        embedding_numpy = np.array(self.soft_embedding.weight.data)
        with open(file, "wb") as f:
            pickle.dump(embedding_numpy, f)