from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GuidebiasArguments
from torch.nn.parameter import Parameter
from torch.optim.adamw import AdamW
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


def prepare_models_and_tokenizer(args: GuidebiasArguments) -> Tuple[BertForMaskedLM, BertForMaskedLM, BertTokenizer]:
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


class JSDivergenceForTriplet(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            dim (int, optional): A dimension along which softmax will be computed. Defaults to 1.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()
        self.reduction = reduction
        self.pi = Parameter(torch.rand(3).softmax(-1))

    def forward(self, dist_p: torch.Tensor, dist_q: torch.Tensor, dist_r: torch.Tensor) -> torch.Tensor:
        self.pi = self.pi.softmax(-1)

        dist_p = F.softmax(dist_p, dim=1)
        dist_q = F.softmax(dist_q, dim=1)
        dist_r = F.softmax(dist_r, dim=1)

        dist_avg = self.pi[0] * dist_p + self.pi[1] * dist_q + self.pi[2] * dist_r

        jsd = 0.0
        jsd += self.pi[0] * F.kl_div(input=F.log_softmax(dist_p, dim=1), target=dist_avg, reduction=self.reduction)
        jsd += self.pi[1] * F.kl_div(input=F.log_softmax(dist_q, dim=1), target=dist_avg, reduction=self.reduction)
        jsd += self.pi[2] * F.kl_div(input=F.log_softmax(dist_r, dim=1), target=dist_avg, reduction=self.reduction)

        return jsd


class UncertaintyLoss(nn.Module):
    def __init__(self, num_losses: int) -> None:
        super().__init__()
        self.noise = Parameter(torch.rand(num_losses).softmax(-1))
        self.num_losses = num_losses

    def forward(self, *loss_fn) -> torch.Tensor:
        loss = 0
        for i in range(self.num_losses):
            loss += loss_fn[i] / (2 * self.noise[i] ** 2)
        loss += torch.log(self.noise.pow(2).prod())

        return loss


def get_hidden_states(
    teacher_neutral_outputs: MaskedLMOutput,
    christian_outputs: MaskedLMOutput,
    jewish_outputs: MaskedLMOutput,
    muslim_outputs: MaskedLMOutput,
    student_neutral_outputs: MaskedLMOutput,
    layer_no: int,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_neutral_hidden = teacher_neutral_outputs.hidden_states[layer_no].mean(dim)
    christian_hidden = christian_outputs.hidden_states[layer_no].mean(dim)
    jewish_hidden = jewish_outputs.hidden_states[layer_no].mean(dim)
    muslim_hidden = muslim_outputs.hidden_states[layer_no].mean(dim)
    student_neutral_hidden = student_neutral_outputs.hidden_states[layer_no].mean(dim)

    return teacher_neutral_hidden, christian_hidden, jewish_hidden, muslim_hidden, student_neutral_hidden


class Runner(pl.LightningModule):
    def __init__(
        self,
        teacher: BertForMaskedLM,
        student: BertForMaskedLM,
        num_training_steps: int,
        num_warmup_steps: int,
        args: GuidebiasArguments,
    ) -> None:
        super().__init__()
        self.teacher = teacher.requires_grad_(False)
        self.student = student
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.js_div_triplet = JSDivergenceForTriplet()
        self.criterion = UncertaintyLoss(2)
        self.lambda_lm = Parameter(torch.rand(2).softmax(-1))
        self.lambda_bias = Parameter(torch.rand(2).softmax(-1))
        self.args = args

    def _get_outputs(
        self, batch: dict
    ) -> Tuple[MaskedLMOutput, MaskedLMOutput, MaskedLMOutput, MaskedLMOutput, MaskedLMOutput]:
        with torch.no_grad():
            teacher_neutral_outputs: MaskedLMOutput = self.teacher.forward(
                input_ids=batch["neutral_input_ids"],
                attention_mask=batch["neutral_attention_mask"],
                token_type_ids=batch["neutral_token_type_ids"],
                output_hidden_states=True,
            )
        christian_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["christian_input_ids"],
            attention_mask=batch["christian_attention_mask"],
            token_type_ids=batch["christian_token_type_ids"],
            output_hidden_states=True,
        )
        jewish_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["jewish_input_ids"],
            attention_mask=batch["jewish_attention_mask"],
            token_type_ids=batch["jewish_token_type_ids"],
            output_hidden_states=True,
        )
        muslim_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["muslim_input_ids"],
            attention_mask=batch["muslim_attention_mask"],
            token_type_ids=batch["muslim_token_type_ids"],
            output_hidden_states=True,
        )
        student_neutral_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["neutral_input_ids"],
            attention_mask=batch["neutral_attention_mask"],
            token_type_ids=batch["neutral_token_type_ids"],
            output_hidden_states=True,
        )

        return teacher_neutral_outputs, christian_outputs, jewish_outputs, muslim_outputs, student_neutral_outputs

    def _get_bias_loss(self, dist_p: torch.Tensor, dist_q: torch.Tensor, dist_r: torch.Tensor) -> torch.Tensor:
        self.lambda_bias = self.lambda_bias.softmax(-1)
        jsd = self.js_div_triplet.forward(dist_p=dist_p, dist_q=dist_q, dist_r=dist_r)
        cosd = (
            1
            - torch.stack(
                [
                    F.cosine_similarity(dist_p, dist_q).mean(),
                    F.cosine_similarity(dist_q, dist_r).mean(),
                    F.cosine_similarity(dist_r, dist_p).mean(),
                ]
            ).mean()
        )

        return self.lambda_bias[0] * jsd + self.lambda_bias[1] * cosd

    def _get_lm_loss(self, dist_p: torch.Tensor, dist_q: torch.Tensor) -> torch.Tensor:
        self.lambda_lm = self.lambda_lm.softmax(-1)
        kld = F.kl_div(input=F.log_softmax(dist_p, dim=-1), target=F.softmax(dist_q, dim=-1), reduction="batchmean")
        cosd = 1 - F.cosine_similarity(dist_p, dist_q).mean()

        return self.lambda_lm[0] * kld + self.lambda_lm[1] * cosd

    def training_step(self, batch) -> torch.Tensor:
        (
            teacher_neutral_outputs,
            christian_outputs,
            jewish_outputs,
            muslim_outputs,
            student_neutral_outputs,
        ) = self._get_outputs(batch)

        (
            teacher_neutral_hidden,
            christian_hidden,
            jewish_hidden,
            muslim_hidden,
            student_neutral_hidden,
        ) = get_hidden_states(
            teacher_neutral_outputs=teacher_neutral_outputs,
            christian_outputs=christian_outputs,
            jewish_outputs=jewish_outputs,
            muslim_outputs=muslim_outputs,
            student_neutral_outputs=student_neutral_outputs,
            layer_no=-1,
            dim=1,
        )

        bias_loss = self._get_bias_loss(dist_p=christian_hidden, dist_q=jewish_hidden, dist_r=muslim_hidden)
        lm_loss = self._get_lm_loss(dist_p=student_neutral_hidden, dist_q=teacher_neutral_hidden)
        loss = self.criterion.forward(bias_loss, lm_loss)

        self.log(name="enc-religion-loss", value=loss)

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for name, p in self.student.named_parameters() if not any(nd in name for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {"params": self.criterion.parameters(), "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=self.num_warmup_steps,
                    num_training_steps=self.num_training_steps,
                ),
                "interval": "step",
            },
        }
