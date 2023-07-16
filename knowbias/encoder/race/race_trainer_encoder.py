import pytorch_lightning as pl
from config import GuidebiasArguments
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from race_dataloader_encoder import prepare_dataloader
from race_modeling_encoder import GuidebiasRunner, prepare_models_and_tokenizer
from transformers.hf_argparser import HfArgumentParser


def train_encoder_for_race(args: GuidebiasArguments):
    #
    wandb_logger = WandbLogger(name=args.bias_type + "-" + args.run_name, project=args.project, entity="squiduu")
    #
    seed_everything(args.seed)

    #
    guide, trainee, tokenizer = prepare_models_and_tokenizer(args)
    #
    train_dataloader = prepare_dataloader(tokenizer=tokenizer, args=args)

    #
    num_training_steps = int(args.num_epochs * len(train_dataloader) / args.num_gpus)
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)

    #
    guidebias_runner = GuidebiasRunner(
        teacher=guide,
        student=trainee,
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
    parser = HfArgumentParser((GuidebiasArguments))
    args = parser.parse_args_into_dataclasses()[0]

    train_encoder_for_race(args)
