from transformers import BertForSequenceClassification
from typing import Union, Tuple
import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self, num_labels: int, freeze: bool = False) -> None:
        """
        The BERT model's class.

        Args:
            num_labels (int): the number of labels.
            freeze (bool): whether freeze the BERT model's parameters or not.
                Defaults to False.
        """
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
        )

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        target: Union[torch.Tensor, None],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function responsible for model's forward step
        (used during the training and the inference).

        Args:
            input_ids (torch.Tensor): the input ids tensor.
            attention_masks (torch.Tensor): the attention masks tensor.
            target (Union[torch.Tensor, None]): the target/label tensor.
                The value must be None if we are making an inference.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the loss and logits, respectively.
                If we are making an inference, then we only return the logits.
        """
        if target is not None:
            output = self.model(
                input_ids=input_ids,
                token_type_ids=None,
                attention_mask=attention_masks,
                labels=target,
                return_dict=None,
            )
            return output["loss"], output["logits"]

        output = self.model(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attention_masks,
            return_dict=None,
        )
        return output["logits"]
