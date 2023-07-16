from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GuidebiasArguments
from torch.optim.adamw import AdamW
from transformers.models.bert.modeling_bert import BertForMaskedLM, BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup


def prepare_models_and_tokenizer(args: GuidebiasArguments) -> Tuple[BertForMaskedLM, BertForMaskedLM, BertTokenizer]:
    # get tokenizer regardless of model version
    tokenizer = BertTokenizer.from_pretrained(args.model_id)
    teacher = BertForMaskedLM.from_pretrained(args.model_id)
    teacher_encoder = BertModel.from_pretrained(args.model_id)
    student = BertForMaskedLM.from_pretrained(args.model_id)
    student_encoder = BertModel.from_pretrained(args.student_encoder_id)

    #
    teacher.bert = teacher_encoder
    teacher.requires_grad_(False)
    #
    student.bert = student_encoder
    student.bert.requires_grad_(False)

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

    def forward(self, logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
        logits_p = F.softmax(logits_p, dim=1)
        logits_q = F.softmax(logits_q, dim=1)

        logits_avg = (logits_p + logits_q) / 2.0

        jsd = 0.0
        jsd += F.kl_div(input=F.log_softmax(logits_p, dim=1), target=logits_avg, reduction=self.reduction)
        jsd += F.kl_div(input=F.log_softmax(logits_q, dim=1), target=logits_avg, reduction=self.reduction)

        return jsd / 2.0


class DivGuidebias(pl.LightningModule):
    def __init__(
        self,
        tokenizer: BertTokenizer,
        teacher: BertForMaskedLM,
        student: BertForMaskedLM,
        asian_ids: List[int],
        black_ids: List[int],
        caucasian_ids: List[int],
        neutral_ids: List[str],
        num_training_steps: int,
        num_warmup_steps: int,
        args: GuidebiasArguments,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.teacher = teacher
        self.student = student
        self.asian_ids = asian_ids
        self.black_ids = black_ids
        self.caucasian_ids = caucasian_ids
        self.neutral_ids = neutral_ids
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.args = args
        self.js_div = JSDivergence("batchmean")

    def _get_logits(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autograd.grad_mode.no_grad():
            teacher_stereo_logits = self.teacher.forward(
                input_ids=batch["stereo_input_ids"],
                attention_mask=batch["stereo_attention_mask"],
                token_type_ids=batch["stereo_token_type_ids"],
            ).logits
            teacher_neutral_logits = self.teacher.forward(
                input_ids=batch["neutral_input_ids"],
                attention_mask=batch["neutral_attention_mask"],
                token_type_ids=batch["neutral_token_type_ids"],
            ).logits

        student_stereo_logits = self.student.forward(
            input_ids=batch["stereo_input_ids"],
            attention_mask=batch["stereo_attention_mask"],
            token_type_ids=batch["stereo_token_type_ids"],
        ).logits
        student_neutral_logits = self.student.forward(
            input_ids=batch["neutral_input_ids"],
            attention_mask=batch["neutral_attention_mask"],
            token_type_ids=batch["neutral_token_type_ids"],
        ).logits

        return teacher_stereo_logits, teacher_neutral_logits, student_stereo_logits, student_neutral_logits

    def _get_divided_logits(
        self,
        teacher_stereo_logits: torch.Tensor,
        teacher_neutral_logits: torch.Tensor,
        student_stereo_logits: torch.Tensor,
        student_neutral_logits: torch.Tensor,
        batch: dict,
    ):
        teacher_stereo_logits = teacher_stereo_logits[
            torch.arange(torch.Tensor.size(batch["stereo_input_ids"])[0]), batch["mask_pos"]
        ][:, self.neutral_ids]
        teacher_neutral_logits = teacher_neutral_logits[
            torch.arange(torch.Tensor.size(batch["neutral_input_ids"])[0]), batch["mask_pos"], :
        ]
        student_asian_logits = student_stereo_logits[
            torch.arange(torch.Tensor.size(batch["stereo_input_ids"])[0]), batch["mask_pos"]
        ][:, self.asian_ids]
        student_black_logits = student_stereo_logits[
            torch.arange(torch.Tensor.size(batch["stereo_input_ids"])[0]), batch["mask_pos"]
        ][:, self.black_ids]
        student_caucasian_logits = student_stereo_logits[
            torch.arange(torch.Tensor.size(batch["stereo_input_ids"])[0]), batch["mask_pos"]
        ][:, self.caucasian_ids]
        student_stereo_logits = student_stereo_logits[
            torch.arange(torch.Tensor.size(batch["stereo_input_ids"])[0]), batch["mask_pos"]
        ][:, self.neutral_ids]
        student_neutral_logits = student_neutral_logits[
            torch.arange(torch.Tensor.size(batch["neutral_input_ids"])[0]), batch["mask_pos"], :
        ]

        return (
            teacher_stereo_logits,
            teacher_neutral_logits,
            student_asian_logits,
            student_black_logits,
            student_caucasian_logits,
            student_stereo_logits,
            student_neutral_logits,
        )

    def _get_bias_loss(self, student_asian_logits, student_black_logits, student_caucasian_logits) -> torch.Tensor:
        js_div_pq = self.js_div.forward(logits_p=student_asian_logits, logits_q=student_black_logits)
        js_div_qr = self.js_div.forward(logits_p=student_black_logits, logits_q=student_caucasian_logits)
        js_div_rp = self.js_div.forward(logits_p=student_caucasian_logits, logits_q=student_asian_logits)

        return js_div_pq + js_div_qr + js_div_rp

    def _get_lm_loss(
        self, teacher_stereo_logits, teacher_neutral_logits, student_stereo_logits, student_neutral_logits
    ) -> torch.Tensor:
        kl_div_stereo = F.kl_div(
            input=F.log_softmax(student_stereo_logits, dim=1),
            target=F.softmax(teacher_stereo_logits, dim=1),
            reduction="batchmean",
        )
        kl_div_neutral = F.kl_div(
            input=F.log_softmax(student_neutral_logits, dim=1),
            target=F.softmax(teacher_neutral_logits, dim=1),
            reduction="batchmean",
        )

        return kl_div_stereo + kl_div_neutral

    def training_step(self, batch) -> torch.Tensor:
        teacher_stereo_logits, teacher_neutral_logits, student_stereo_logits, student_neutral_logits = self._get_logits(
            batch
        )
        (
            teacher_stereo_logits,
            teacher_neutral_logits,
            student_asian_logits,
            student_black_logits,
            student_caucasian_logits,
            student_stereo_logits,
            student_neutral_logits,
        ) = self._get_divided_logits(
            teacher_stereo_logits=teacher_stereo_logits,
            teacher_neutral_logits=teacher_neutral_logits,
            student_stereo_logits=student_stereo_logits,
            student_neutral_logits=student_neutral_logits,
            batch=batch,
        )

        bias_loss = self._get_bias_loss(
            student_asian_logits=student_asian_logits,
            student_black_logits=student_black_logits,
            student_caucasian_logits=student_caucasian_logits,
        )
        lm_loss = self._get_lm_loss(
            teacher_stereo_logits=teacher_stereo_logits,
            teacher_neutral_logits=teacher_neutral_logits,
            student_stereo_logits=student_stereo_logits,
            student_neutral_logits=student_neutral_logits,
        )
        loss = bias_loss + lm_loss

        self.log(name="race-mlm-loss", value=loss)

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [p for n, p in self.student.cls.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.student.cls.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params=optimizer_grouped_params, lr=self.args.lr)

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
