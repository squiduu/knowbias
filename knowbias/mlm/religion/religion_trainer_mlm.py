import pytorch_lightning as pl
from config import CustomArguments
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from religion_dataloader_mlm import prepare_dataloader, read_data
from religion_modeling_mlm import Runner, prepare_models_and_tokenizer
from transformers.hf_argparser import HfArgumentParser


def run_race_trainer(args: CustomArguments):
    wandb_logger = (
        WandbLogger(
            name=args.bias_type
            + "_mlm"
            + "_sd"
            + str(args.seed)
            + "_ep"
            + str(args.num_epochs)
            + "_wk"
            + str(args.num_wiki_words)
            + "_bs"
            + str(args.batch_size),
            project=args.project,
            entity="squiduu",
        )
        if args.use_wandb
        else False
    )
    seed_everything(args.seed)
    #
    teacher, student, tokenizer = prepare_models_and_tokenizer(args)

    #
    christian_ids, jewish_ids, muslim_ids, neutral_ids, dataset = read_data(tokenizer=tokenizer, args=args)
    dataloader = prepare_dataloader(dataset=dataset, tokenizer=tokenizer, args=args)

    #
    num_training_steps = int(args.num_epochs * len(dataloader))
    num_warmup_steps = int(num_training_steps * args.warmup_proportion)
    runner = Runner(
        tokenizer=tokenizer,
        teacher=teacher,
        student=student,
        christian_ids=christian_ids,
        jewish_ids=jewish_ids,
        muslim_ids=muslim_ids,
        neutral_ids=neutral_ids,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
        args=args,
    )

    trainer = pl.Trainer(
        logger=wandb_logger if wandb_logger else False,
        enable_checkpointing=False,
        gradient_clip_val=1.0,
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy="ddp",
        precision=args.precision,
    )
    trainer.fit(model=runner, train_dataloaders=dataloader)

    student.save_pretrained(f"./out/{args.run_name}")


if __name__ == "__main__":
    parser = HfArgumentParser(CustomArguments)
    args = parser.parse_args_into_dataclasses()[0]

    run_race_trainer(args)
