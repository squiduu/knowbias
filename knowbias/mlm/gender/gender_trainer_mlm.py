import pytorch_lightning as pl
from config import GuidebiasArguments
from gender_dataloader_mlm import prepare_dataloader, read_data
from gender_modeling_mlm import DivGuidebias, prepare_models_and_tokenizer
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from transformers.hf_argparser import HfArgumentParser


def run_gender_trainer(args: GuidebiasArguments):
    #
    wandb_logger = WandbLogger(
        name=args.bias_type
        + "_mlm"
        + "_sd"
        + str(args.seed)
        + "_ep"
        + str(args.num_epochs)
        + "_wk"
        + str(args.num_wiki_words)
        + "_tg"
        + str(args.num_target_words)
        + "_bs"
        + str(args.batch_size)
        + "_lr"
        + str(args.lr)[0],
        project=args.project,
        entity="squiduu",
    )

    #
    seed_everything(args.seed)

    #
    teacher, student, tokenizer = prepare_models_and_tokenizer(args)

    #
    male_ids, female_ids, neutral_ids, dataset = read_data(tokenizer=tokenizer, args=args)
    train_dataloader = prepare_dataloader(dataset=dataset, args=args)

    #
    num_training_steps = int(args.num_epochs * len(train_dataloader))
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)

    #
    div_guidebias = DivGuidebias(
        tokenizer=tokenizer,
        teacher=teacher,
        student=student,
        male_ids=male_ids,
        female_ids=female_ids,
        neutral_ids=neutral_ids,
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
        precision=args.precision,
    )
    trainer.fit(model=div_guidebias, train_dataloaders=train_dataloader)

    #
    div_guidebias.student.save_pretrained(
        f"./out/{args.bias_type}_mlm_sd{str(args.seed)}_ep{str(args.num_epochs)}_wk{str(args.num_wiki_words)}_tg{str(args.num_target_words)}_bs{str(args.batch_size)}_lr{str(args.lr)[0]}"
    )


if __name__ == "__main__":
    parser = HfArgumentParser((GuidebiasArguments))
    args = parser.parse_args_into_dataclasses()[0]

    run_gender_trainer(args)
