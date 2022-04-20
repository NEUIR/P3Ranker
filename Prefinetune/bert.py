from typing import Tuple

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel

class Bert(nn.Module):
    def __init__(
        self,
        pretrained: str,
        num_labels:int =3
    ) -> None:
        super(Bert, self).__init__()
        self._pretrained = pretrained

        self._config = AutoConfig.from_pretrained(self._pretrained)
        self._model = AutoModel.from_pretrained(self._pretrained, config=self._config)

        self._dense = nn.Linear(self._config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self._model(input_ids, attention_mask = attention_mask,return_dict=False)
        logits = output[0][:, 0, :]
        score = self._dense(logits).squeeze(-1)
        return score, logits
