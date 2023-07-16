import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from config import GuidebiasArguments
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer


class CustomDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


def get_religion_words(args: GuidebiasArguments) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    with open(file=f"../../data/target/religion/christian.json", mode="r") as christian_fp:
        christian_words = json.load(christian_fp)
    with open(file=f"../../data/target/religion/jewish.json", mode="r") as jewish_fp:
        jewish_words = json.load(jewish_fp)
    with open(file="../../data/target/religion/muslim.json", mode="r") as muslim_fp:
        muslim_words = json.load(muslim_fp)

    with open(file=f"../../data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        stereo_words = json.load(ster_fp)

    with open(file=f"../../data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        wiki_words = json.load(wiki_fp)
    wiki_words = filter_wiki_words(
        wiki_words=wiki_words, target_words=christian_words + jewish_words + muslim_words, stereo_words=stereo_words
    )
    wiki_words = wiki_words[: args.num_wiki_words]

    return christian_words, jewish_words, muslim_words, stereo_words, wiki_words


def filter_wiki_words(wiki_words: List[str], target_words: List[str], stereo_words: List[str]) -> List[str]:
    filtered = []
    for wiki_word in wiki_words:
        if wiki_word not in target_words + stereo_words:
            filtered.append(wiki_word)

    return filtered


def prepare_stereo_sents(target_words: List[str], wiki_words: List[str], stereo_words: List[str]) -> List[str]:
    sents = []
    for target_word in target_words:
        for wiki_word in wiki_words:
            for stereo_word in stereo_words:
                sents.append(target_word + " " + wiki_word + " " + stereo_word + " .")

    return sents


def prepare_neutral_sents(target_words: List[str], wiki_words: List[str]) -> List[str]:
    sents = []
    for target_word in target_words:
        for wiki_word_0 in wiki_words:
            for wiki_word_1 in wiki_words:
                sents.append(target_word + " " + wiki_word_0 + " " + wiki_word_1 + " .")

    return sents


def read_data(tokenizer: BertTokenizer, args: GuidebiasArguments) -> List[Dict[str, torch.Tensor]]:
    dataset = []

    christian_words, jewish_words, muslim_words, stereo_words, wiki_words = get_religion_words(args)

    christian_sents = prepare_stereo_sents(
        target_words=christian_words, wiki_words=wiki_words[: args.num_stereo_words], stereo_words=stereo_words
    )
    jewish_sents = prepare_stereo_sents(
        target_words=jewish_words, wiki_words=wiki_words[: args.num_stereo_words], stereo_words=stereo_words
    )
    muslim_sents = prepare_stereo_sents(
        target_words=muslim_words, wiki_words=wiki_words[: args.num_stereo_words], stereo_words=stereo_words
    )
    neutral_sents = prepare_neutral_sents(
        target_words=christian_words + jewish_words + muslim_words, wiki_words=wiki_words
    )

    if len(christian_sents) >= len(neutral_sents):
        indices = np.random.choice(len(christian_sents), size=len(neutral_sents), replace=False)

        christian_sents_selected = []
        jewish_sents_selected = []
        muslim_sents_selected = []
        for i in indices:
            christian_sents_selected.append(christian_sents[i])
            jewish_sents_selected.append(jewish_sents[i])
            muslim_sents_selected.append(muslim_sents[i])

        christian_inputs = tokenizer.__call__(
            text=christian_sents_selected, padding=True, return_tensors="pt", truncation=True
        )
        jewish_inputs = tokenizer.__call__(
            text=jewish_sents_selected, padding=True, return_tensors="pt", truncation=True
        )
        muslim_inputs = tokenizer.__call__(
            text=muslim_sents_selected, padding=True, return_tensors="pt", truncation=True
        )
        neutral_inputs = tokenizer.__call__(text=neutral_sents, padding=True, return_tensors="pt", truncation=True)

    else:
        neutral_sents = np.random.choice(neutral_sents, size=len(christian_sents), replace=False).tolist()

        christian_inputs = tokenizer.__call__(text=christian_sents, padding=True, return_tensors="pt", truncation=True)
        jewish_inputs = tokenizer.__call__(text=jewish_sents, padding=True, return_tensors="pt", truncation=True)
        muslim_inputs = tokenizer.__call__(text=muslim_sents, padding=True, return_tensors="pt", truncation=True)
        neutral_inputs = tokenizer.__call__(text=neutral_sents, padding=True, return_tensors="pt", truncation=True)

    for i in range(torch.Tensor.size(christian_inputs["input_ids"])[0]):
        dataset.append(
            {
                "christian_input_ids": christian_inputs["input_ids"][i],
                "christian_attention_mask": christian_inputs["attention_mask"][i],
                "christian_token_type_ids": christian_inputs["token_type_ids"][i],
                "jewish_input_ids": jewish_inputs["input_ids"][i],
                "jewish_attention_mask": jewish_inputs["attention_mask"][i],
                "jewish_token_type_ids": jewish_inputs["token_type_ids"][i],
                "muslim_input_ids": muslim_inputs["input_ids"][i],
                "muslim_attention_mask": muslim_inputs["attention_mask"][i],
                "muslim_token_type_ids": muslim_inputs["token_type_ids"][i],
                "neutral_input_ids": neutral_inputs["input_ids"][i],
                "neutral_attention_mask": neutral_inputs["attention_mask"][i],
                "neutral_token_type_ids": neutral_inputs["token_type_ids"][i],
            }
        )

    return dataset


def prepare_dataloader(tokenizer: BertTokenizer, args: GuidebiasArguments) -> DataLoader:
    #
    train_dataset = CustomDataset(read_data(tokenizer=tokenizer, args=args))

    #
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_dataloader
