from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GuidebiasArguments
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


class JSDivergence(nn.Module):
    def __init__(self, reduction: str = "batchmean") -> None:
        """Get average JS-Divergence between two networks.

        Args:
            dim (int, optional): A dimension along which softmax will be computed. Defaults to 1.
            reduction (str, optional): Specifies the reduction to apply to the output. Defaults to "batchmean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, distribution_p: torch.Tensor, distribution_q: torch.Tensor) -> torch.Tensor:
        distribution_p = F.softmax(distribution_p, dim=1)
        distribution_q = F.softmax(distribution_q, dim=1)

        distribution_avg = (distribution_p + distribution_q) / 2.0

        jsd = 0.0
        jsd += F.kl_div(input=F.log_softmax(distribution_p, dim=1), target=distribution_avg, reduction=self.reduction)
        jsd += F.kl_div(input=F.log_softmax(distribution_q, dim=1), target=distribution_avg, reduction=self.reduction)

        return jsd / 2.0


def get_hidden_states(
    teacher_neutral_outputs: MaskedLMOutput,
    asian_outputs: MaskedLMOutput,
    black_outputs: MaskedLMOutput,
    caucasian_outputs: MaskedLMOutput,
    student_neutral_outputs: MaskedLMOutput,
    layer_no: int,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_neutral_hidden = teacher_neutral_outputs.hidden_states[layer_no].mean(dim)
    asian_hidden = asian_outputs.hidden_states[layer_no].mean(dim)
    black_hidden = black_outputs.hidden_states[layer_no].mean(dim)
    caucasian_hidden = caucasian_outputs.hidden_states[layer_no].mean(dim)
    student_neutral_hidden = student_neutral_outputs.hidden_states[layer_no].mean(dim)

    return teacher_neutral_hidden, asian_hidden, black_hidden, caucasian_hidden, student_neutral_hidden


def get_lm_loss(distribution_p: torch.Tensor, distribution_q: torch.Tensor) -> torch.Tensor:
    kld = F.kl_div(
        input=F.log_softmax(distribution_p, dim=-1), target=F.softmax(distribution_q, dim=-1), reduction="batchmean"
    )
    cossim = F.cosine_similarity(distribution_p, distribution_q).mean()

    return kld - cossim


class GuidebiasRunner(pl.LightningModule):
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
        self.args = args
        self.js_div = JSDivergence()

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
        asian_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["asian_input_ids"],
            attention_mask=batch["asian_attention_mask"],
            token_type_ids=batch["asian_token_type_ids"],
            output_hidden_states=True,
        )
        black_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["black_input_ids"],
            attention_mask=batch["black_attention_mask"],
            token_type_ids=batch["black_token_type_ids"],
            output_hidden_states=True,
        )
        caucasian_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["caucasian_input_ids"],
            attention_mask=batch["caucasian_attention_mask"],
            token_type_ids=batch["caucasian_token_type_ids"],
            output_hidden_states=True,
        )
        student_neutral_outputs: MaskedLMOutput = self.student.forward(
            input_ids=batch["neutral_input_ids"],
            attention_mask=batch["neutral_attention_mask"],
            token_type_ids=batch["neutral_token_type_ids"],
            output_hidden_states=True,
        )

        return teacher_neutral_outputs, asian_outputs, black_outputs, caucasian_outputs, student_neutral_outputs

    def _get_bias_loss(
        self, distribution_p: torch.Tensor, distribution_q: torch.Tensor, distribution_r: torch.Tensor
    ) -> torch.Tensor:
        js_div_pq = self.js_div.forward(distribution_p=distribution_p, distribution_q=distribution_q)
        js_div_qr = self.js_div.forward(distribution_p=distribution_q, distribution_q=distribution_r)
        js_div_rp = self.js_div.forward(distribution_p=distribution_r, distribution_q=distribution_p)

        cossim_pq = F.cosine_similarity(distribution_p, distribution_q).mean()
        cossim_qr = F.cosine_similarity(distribution_q, distribution_r).mean()
        cossim_rp = F.cosine_similarity(distribution_r, distribution_p).mean()

        return js_div_pq + js_div_qr + js_div_rp - cossim_pq - cossim_qr - cossim_rp

    def training_step(self, batch) -> torch.Tensor:
        (
            teacher_neutral_outputs,
            asian_outputs,
            black_outputs,
            caucasian_outputs,
            student_neutral_outputs,
        ) = self._get_outputs(batch)

        (
            teacher_neutral_hidden,
            asian_hidden,
            black_hidden,
            caucasian_hidden,
            student_neutral_hidden,
        ) = get_hidden_states(
            teacher_neutral_outputs=teacher_neutral_outputs,
            asian_outputs=asian_outputs,
            black_outputs=black_outputs,
            caucasian_outputs=caucasian_outputs,
            student_neutral_outputs=student_neutral_outputs,
            layer_no=-1,
            dim=1,
        )

        bias_loss = self._get_bias_loss(
            distribution_p=asian_hidden, distribution_q=black_hidden, distribution_r=caucasian_hidden
        )
        lm_loss = get_lm_loss(distribution_p=student_neutral_hidden, distribution_q=teacher_neutral_hidden)
        loss = self.args.debias_ratio * bias_loss + (1 - self.args.debias_ratio) * lm_loss
        self.log(name="enc-race-loss", value=loss)

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
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
