import pytorch_lightning as pl
import torch
from config import MyArguments
from gender_dataloader_encoder import prepare_dataloader
from gender_modeling_encoder import (
    BiasRemover,
    JSDivergence,
    prepare_models_and_tokenizer,
)
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers.hf_argparser import HfArgumentParser


def run_gender_trainer(args: MyArguments):
    #
    wandb_logger = WandbLogger(
        name=args.bias_type + "-" + args.run_name,
        project=args.project,
        entity="squiduu",
    )

    #
    seed_everything(args.seed)

    #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    guide, trainee, tokenizer = prepare_models_and_tokenizer(args)
    guide.to(device)
    trainee.to(device)

    #
    train_dataloader = prepare_dataloader(tokenizer=tokenizer, args=args)

    #
    num_training_steps = int(args.num_epochs * len(train_dataloader) / args.num_gpus)
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)

    #
    jsd_runner = JSDivergence(reduction="batchmean")

    #
    guidebias_runner = BiasRemover(
        teacher=guide,
        student=trainee,
        jsd_runner=jsd_runner,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        args=args,
    )

    #
    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp",
        precision=16,
    )
    trainer.fit(model=guidebias_runner, train_dataloaders=train_dataloader)

    #
    trainee.save_pretrained(f"./out/{args.bias_type}_{args.run_name}")


if __name__ == "__main__":
    parser = HfArgumentParser(MyArguments)
    args = parser.parse_args_into_dataclasses()

    run_gender_trainer(args)
