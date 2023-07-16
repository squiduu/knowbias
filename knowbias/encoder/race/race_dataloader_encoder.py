import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from config import GuidebiasArguments
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer


class GuidebiasDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


def get_race_words(args: GuidebiasArguments) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    with open(file=f"../../../data/target/race/asian.json", mode="r") as a_fp:
        asian_words = json.load(a_fp)
    with open(file=f"../../../data/target/race/black.json", mode="r") as b_fp:
        black_words = json.load(b_fp)
    with open(file="../../../data/target/race/caucasian.json", mode="r") as c_fp:
        caucasian_words = json.load(c_fp)

    with open(file=f"../../../data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        stereo_words = json.load(ster_fp)

    with open(file=f"../../../data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        wiki_words = json.load(wiki_fp)
    wiki_words = filter_wiki(
        wiki_words=wiki_words, target_words=asian_words + black_words + caucasian_words, stereo_words=stereo_words
    )
    wiki_words = wiki_words[: args.num_wiki_words]

    return asian_words, black_words, caucasian_words, stereo_words, wiki_words


def filter_wiki(wiki_words: List[str], target_words: List[str], stereo_words: List[str]) -> List[str]:
    filtered = []
    for word in wiki_words:
        if word not in (target_words + stereo_words):
            filtered.append(word)

    return filtered


def prepare_stereo_sents(targ_words: List[str], wiki_words: List[str], ster_words: List[str]) -> List[str]:
    sents = []
    for targ_word in targ_words:
        for wiki_word in wiki_words:
            for ster_word in ster_words:
                sents.append(targ_word + " " + wiki_word + " " + ster_word + " .")

    return sents


def prepare_neutral_sents(targ_words: List[str], wiki_words: List[str]) -> List[str]:
    sents = []
    for targ_word in targ_words:
        for wiki_word_0 in wiki_words:
            for wiki_word_1 in wiki_words:
                sents.append(targ_word + " " + wiki_word_0 + " " + wiki_word_1 + " .")

    return sents


def read_data(tokenizer: BertTokenizer, args: GuidebiasArguments) -> List[Dict[str, torch.Tensor]]:
    dataset = []

    asian_words, black_words, caucasian_words, stereo_words, wiki_words = get_race_words(args)

    asian_sents = prepare_stereo_sents(
        targ_words=asian_words, wiki_words=wiki_words[: args.num_stereo_words], ster_words=stereo_words
    )
    black_sents = prepare_stereo_sents(
        targ_words=black_words, wiki_words=wiki_words[: args.num_stereo_words], ster_words=stereo_words
    )
    caucasian_sents = prepare_stereo_sents(
        targ_words=caucasian_words, wiki_words=wiki_words[: args.num_stereo_words], ster_words=stereo_words
    )
    neutral_sents = prepare_neutral_sents(targ_words=asian_words + black_words + caucasian_words, wiki_words=wiki_words)

    if len(asian_sents) >= len(neutral_sents):
        indices = np.random.choice(len(asian_sents), size=len(neutral_sents), replace=False)

        asian_sents_selected = []
        black_sents_selected = []
        caucasian_sents_selected = []
        for i in indices:
            asian_sents_selected.append(asian_sents[i])
            black_sents_selected.append(black_sents[i])
            caucasian_sents_selected.append(caucasian_sents[i])

        asian_inputs = tokenizer.__call__(text=asian_sents_selected, padding=True, return_tensors="pt", truncation=True)
        black_inputs = tokenizer.__call__(text=black_sents_selected, padding=True, return_tensors="pt", truncation=True)
        caucasian_inputs = tokenizer.__call__(
            text=caucasian_sents_selected, padding=True, return_tensors="pt", truncation=True
        )
        neutral_inputs = tokenizer.__call__(text=neutral_sents, padding=True, return_tensors="pt", truncation=True)

    else:
        neutral_sents = np.random.choice(neutral_sents, size=len(asian_sents), replace=False).tolist()

        asian_inputs = tokenizer.__call__(text=asian_sents, padding=True, return_tensors="pt", truncation=True)
        black_inputs = tokenizer.__call__(text=black_sents, padding=True, return_tensors="pt", truncation=True)
        caucasian_inputs = tokenizer.__call__(text=caucasian_sents, padding=True, return_tensors="pt", truncation=True)
        neutral_inputs = tokenizer.__call__(text=neutral_sents, padding=True, return_tensors="pt", truncation=True)

    for i in range(torch.Tensor.size(asian_inputs["input_ids"])[0]):
        dataset.append(
            {
                "asian_input_ids": asian_inputs["input_ids"][i],
                "asian_attention_mask": asian_inputs["attention_mask"][i],
                "asian_token_type_ids": asian_inputs["token_type_ids"][i],
                "black_input_ids": black_inputs["input_ids"][i],
                "black_attention_mask": black_inputs["attention_mask"][i],
                "black_token_type_ids": black_inputs["token_type_ids"][i],
                "caucasian_input_ids": caucasian_inputs["input_ids"][i],
                "caucasian_attention_mask": caucasian_inputs["attention_mask"][i],
                "caucasian_token_type_ids": caucasian_inputs["token_type_ids"][i],
                "neutral_input_ids": neutral_inputs["input_ids"][i],
                "neutral_attention_mask": neutral_inputs["attention_mask"][i],
                "neutral_token_type_ids": neutral_inputs["token_type_ids"][i],
            }
        )

    return dataset


def prepare_dataloader(tokenizer: BertTokenizer, args: GuidebiasArguments) -> DataLoader:
    #
    train_dataset = GuidebiasDataset(read_data(tokenizer=tokenizer, args=args))

    #
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_dataloader
