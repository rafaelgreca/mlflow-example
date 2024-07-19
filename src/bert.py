from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from typing import Union, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn as nn


def train_bert(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
    scheduler: get_linear_schedule_with_warmup,
) -> Tuple[float, float]:
    """
    Function responsible for the training step.

    Args:
        model (torch.nn.Module): the Bert model.
        optimizer (torch.nn.optim.AdamW): the optimizer that will be used.
        device (torch.device): the torch device (cpu or cuda).
        dataloader (torch.utils.data.DataLoader): the training dataloader.
        scheduler (get_linear_schedule_with_warmup): BERT's warmup scheduler.

    Returns:
        Tuple[Dict, float]: current epoch metrics score and loss, respectively.
    """
    model.train()
    train_loss = 0
    train_f1 = 0
    train_precision = 0
    train_recall = 0
    train_metrics = {}

    for batch in dataloader:
        input_id, attention_mask, target = batch
        input_id, attention_mask, target = (
            input_id.to(device),
            attention_mask.to(device),
            target.to(device),
        )
        target = target.float()

        optimizer.zero_grad()

        # define inputs
        loss, logits = model(input_id, attention_mask, target)
        train_loss += loss.item()
        loss.backward()

        optimizer.step()

        scheduler.step()

        # getting the index (class) with the highest value
        prediction = torch.argmax(logits, axis=1).flatten()
        target = torch.argmax(target, axis=1).flatten()

        train_f1 += f1_score(
            target.detach().cpu().data.numpy(),
            prediction.detach().cpu().numpy(),
            average="weighted",
            zero_division=0.0,
        )

        train_precision += precision_score(
            target.detach().cpu().data.numpy(),
            prediction.detach().cpu().numpy(),
            average="weighted",
            zero_division=0.0,
        )

        train_recall += recall_score(
            target.detach().cpu().data.numpy(),
            prediction.detach().cpu().numpy(),
            average="weighted",
            zero_division=0.0,
        )

    train_loss /= len(dataloader)
    train_metrics["train f1 score"] = train_f1 / len(dataloader)
    train_metrics["train recall"] = train_recall / len(dataloader)
    train_metrics["train precision"] = train_precision / len(dataloader)
    return train_metrics, train_loss


def test_bert(
    model: torch.nn.Module,
    device: torch.device,
    dataloader: torch.utils.data.DataLoader,
):
    """
    Function responsible for the test/validation step.

    Args:
        model (torch.nn.Module): the Bert model.
        device (torch.device): the torch device (cpu or cuda).
        dataloader (torch.utils.data.DataLoader): the validation dataloader.

    Returns:
        Tuple[Dict, float]: current epoch metrics score and loss, respectively.
    """
    model.eval()
    test_loss = 0
    test_f1 = 0
    test_precision = 0
    test_recall = 0
    test_metrics = {}

    with torch.inference_mode():
        for batch in dataloader:
            input_id, attention_mask, target = batch
            input_id, attention_mask, target = (
                input_id.to(device),
                attention_mask.to(device),
                target.to(device),
            )
            target = target.float()
            loss, logits = model(input_id, attention_mask, target)

            test_loss += loss.item()

            # getting the index (class) with the highest value
            prediction = torch.argmax(logits, axis=1).flatten()
            target = torch.argmax(target, axis=1).flatten()

            test_f1 += f1_score(
                target.detach().cpu().data.numpy(),
                prediction.detach().cpu().numpy(),
                average="weighted",
                zero_division=0.0,
            )

            test_precision += precision_score(
                target.detach().cpu().data.numpy(),
                prediction.detach().cpu().numpy(),
                average="weighted",
                zero_division=0.0,
            )

            test_recall += recall_score(
                target.detach().cpu().data.numpy(),
                prediction.detach().cpu().numpy(),
                average="weighted",
                zero_division=0.0,
            )

    test_loss /= len(dataloader)
    test_metrics["validation f1 score"] = test_f1 / len(dataloader)
    test_metrics["validation recall"] = test_recall / len(dataloader)
    test_metrics["validation precision"] = test_precision / len(dataloader)
    return test_metrics, test_loss


class BERT(nn.Module):
    def __init__(self, num_labels: int, freeze: bool = False) -> None:
        """
        The BERT model's class.

        Args:
            num_labels (int): the number of labels.
            freeze (bool): whether freeze the BERT model's parameters or not.
                Defaults to False.
        """
        super(BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False,
            force_download=False,
            resume_download=False,
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
