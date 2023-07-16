from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GuidebiasArguments:
    max_seq_len: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_len: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    num_target_words: Optional[int] = field(default=0, metadata={"help": "The number of gender words."})
    num_wiki_words: Optional[int] = field(default=0, metadata={"help": "The number of wiki words."})
    num_stereo_words: Optional[int] = field(default=0, metadata={"help": "The number of wiki words for stereo."})
    bias_type: Optional[str] = field(default="gender", metadata={"help": "The type of bias attributes."})
    model_id: Optional[str] = field(
        default="bert-base-uncased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    model_name: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    num_gpus: Optional[int] = field(default=1, metadata={"help": "The number of GPUs for training."})
    student_encoder_id: Optional[str] = field(
        default="squiduu/guidebias-bert-base-uncased", metadata={"help": "The model ID of debiased encoder."}
    )
    batch_size: Optional[int] = field(default=1, metadata={"help": "The number of batch size per device."})
    project: Optional[str] = field(default=None, metadata={"help": "A project name."})
    run_name: Optional[str] = field(default=None, metadata={"help": "An optional name of each run."})
    seed: Optional[int] = field(default=42, metadata={"help": "A seed number."})
    lr: Optional[float] = field(default=2e-5, metadata={"help": "A learning rate for training."})
    weight_decay: Optional[float] = field(default=0.01, metadata={"help": "The value of weight decay."})
    num_epochs: Optional[int] = field(default=1, metadata={"help": "A maximum number of epochs."})
    num_workers: Optional[int] = field(default=0, metadata={"help": "A number of workers for dataloader."})
    grad_accum_steps: Optional[int] = field(default=1, metadata={"help": "A number of accumulation steps."})
    warmup_proportion: Optional[float] = field(default=0.0, metadata={"help": "A warm-up proportion for scheduler."})
    precision: Optional[int] = field(default=32, metadata={"help": "The precision value for floating points."})
