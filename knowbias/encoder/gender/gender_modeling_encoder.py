from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import MyArguments
from torch.optim.adamw import AdamW
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


def prepare_models_and_tokenizer(
    args: MyArguments,
) -> Tuple[BertForMaskedLM, BertForMaskedLM, BertTokenizer]:
    # get tokenizer regardless of model version
    tokenizer = BertTokenizer.from_pretrained(args.model_id)
    teacher = BertForMaskedLM.from_pretrained(args.model_id)
    teacher_encoder = BertModel.from_pretrained(args.model_id)
    student = BertForMaskedLM.from_pretrained(args.model_id)
    student_encoder = BertModel.from_pretrained(args.model_id)

    #
    teacher.bert = teacher_encoder
    student.bert = student_encoder

    return teacher, student, tokenizer


class JSDivergence(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            dim (int, optional): A dimension along which softmax will be computed. Defaults to 1.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self, distribution_p: torch.Tensor, distribution_q: torch.Tensor
    ) -> torch.Tensor:
        distribution_p = F.softmax(distribution_p, dim=1)
        distribution_q = F.softmax(distribution_q, dim=1)

        distribution_avg = (distribution_p + distribution_q) / 2.0

        jsd = 0.0
        jsd += F.kl_div(
            input=F.log_softmax(distribution_p, dim=1),
            target=distribution_avg,
            reduction=self.reduction,
        )
        jsd += F.kl_div(
            input=F.log_softmax(distribution_q, dim=1),
            target=distribution_avg,
            reduction=self.reduction,
        )

        return jsd / 2.0


def get_outputs(
    teacher: BertForMaskedLM, student: BertForMaskedLM, batch: dict
) -> Tuple[MaskedLMOutput, MaskedLMOutput, MaskedLMOutput, MaskedLMOutput]:
    with torch.no_grad():
        teacher_neutral_outputs: MaskedLMOutput = teacher.forward(
            input_ids=batch["neutral_input_ids"],
            attention_mask=batch["neutral_attention_mask"],
            token_type_ids=batch["neutral_token_type_ids"],
            output_hidden_states=True,
        )
    student_male_outputs: MaskedLMOutput = student.forward(
        input_ids=batch["male_input_ids"],
        attention_mask=batch["male_attention_mask"],
        token_type_ids=batch["male_token_type_ids"],
        output_hidden_states=True,
    )
    student_female_outputs: MaskedLMOutput = student.forward(
        input_ids=batch["female_input_ids"],
        attention_mask=batch["female_attention_mask"],
        token_type_ids=batch["female_token_type_ids"],
        output_hidden_states=True,
    )
    student_neutral_outputs: MaskedLMOutput = student.forward(
        input_ids=batch["neutral_input_ids"],
        attention_mask=batch["neutral_attention_mask"],
        token_type_ids=batch["neutral_token_type_ids"],
        output_hidden_states=True,
    )

    return (
        teacher_neutral_outputs,
        student_male_outputs,
        student_female_outputs,
        student_neutral_outputs,
    )


def get_hidden_states(
    teacher_neutral_outputs: MaskedLMOutput,
    student_male_outputs: MaskedLMOutput,
    student_female_outputs: MaskedLMOutput,
    student_neutral_outputs: MaskedLMOutput,
    layer_no: int,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_neutral_hidden = teacher_neutral_outputs.hidden_states[layer_no].mean(dim)
    student_male_hidden = student_male_outputs.hidden_states[layer_no].mean(dim)
    student_female_hidden = student_female_outputs.hidden_states[layer_no].mean(dim)
    student_neutral_hidden = student_neutral_outputs.hidden_states[layer_no].mean(dim)

    return (
        teacher_neutral_hidden,
        student_male_hidden,
        student_female_hidden,
        student_neutral_hidden,
    )


class UncertaintyLoss(nn.Module):
    def __init__(self, num_losses: int) -> None:
        super().__init__()
        sigma = torch.randn(num_losses)
        self.sigma = nn.Parameter(sigma)
        self.num_losses = num_losses

    def forward(self, *loss_fn) -> torch.Tensor:
        loss = 0
        for i in range(self.num_losses):
            loss += loss_fn[i] / (2 * self.sigma[i] ** 2)
        loss += torch.log(self.sigma.pow(2).prod())

        return loss


def get_bias_loss(
    jsd_runner: JSDivergence, distribution_p: torch.Tensor, distribution_q: torch.Tensor
):
    jsd = jsd_runner.forward(
        distribution_p=distribution_p, distribution_q=distribution_q
    )
    cossim = F.cosine_similarity(distribution_p, distribution_q).mean()

    return jsd - cossim


def get_lm_loss(distribution_p: torch.Tensor, distribution_q: torch.Tensor):
    kld = F.kl_div(
        input=F.log_softmax(distribution_p, dim=-1),
        target=F.softmax(distribution_q, dim=-1),
        reduction="batchmean",
    )
    cossim = F.cosine_similarity(distribution_p, distribution_q).mean()

    return kld - cossim


class BiasRemover(pl.LightningModule):
    def __init__(
        self,
        teacher: BertForMaskedLM,
        student: BertForMaskedLM,
        jsd_runner: JSDivergence,
        num_training_steps: int,
        num_warmup_steps: int,
        args: MyArguments,
    ) -> None:
        super().__init__()
        self.teacher = teacher.requires_grad_(False)
        self.student = student
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.args = args
        self.jsd_runner = jsd_runner

    def training_step(self, batch) -> torch.Tensor:
        pass

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.train_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=self.train_args.lr
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_linear_schedule_with_warmup(
                    optimizer, self.num_warmup_steps, self.num_training_steps
                ),
                "interval": "step",
            },
        }
