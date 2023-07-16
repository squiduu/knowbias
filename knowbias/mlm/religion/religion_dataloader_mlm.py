import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from config import CustomArguments
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


def filter_target_words(
    tokenizer: BertTokenizer, christian_words: List[str], jewish_words: List[str], muslim_words: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    christian_filtered = []
    jewish_filtered = []
    muslim_filtered = []
    for i in range(max(len(christian_words), len(jewish_words), len(muslim_words))):
        if tokenizer.unk_token_id not in tokenizer.convert_tokens_to_ids(
            [christian_words[i], jewish_words[i], muslim_words[i]]
        ):
            christian_filtered.append(christian_words[i])
            jewish_filtered.append(jewish_words[i])
            muslim_filtered.append(muslim_words[i])

    return christian_filtered, jewish_filtered, muslim_filtered


def get_neutral_ids(
    tokenizer: BertTokenizer,
    christian_words: List[str],
    jewish_words: List[str],
    muslim_words: List[str],
    stereo_words: List[str],
) -> List[int]:
    christian_tokens = []
    jewish_tokens = []
    muslim_tokens = []
    stereo_tokens = []
    for word in christian_words:
        christian_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    for word in jewish_words:
        jewish_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    for word in muslim_words:
        muslim_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    for word in stereo_words:
        stereo_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))

    neutral_ids = [
        id
        for token, id in tokenizer.get_vocab().items()
        if token
        not in list(set(christian_tokens))
        + list(set(jewish_tokens))
        + list(set(muslim_tokens))
        + list(set(stereo_tokens))
    ]

    return neutral_ids


def filter_wiki_words(
    tokenizer: BertTokenizer, wiki_words: List[str], target_words: List[str], stereo_words: List[str]
) -> List[str]:
    wiki_filtered = []
    for word in wiki_words:
        if (
            word not in (target_words + stereo_words)
            and tokenizer.convert_tokens_to_ids(word) != tokenizer.unk_token_id
        ):
            wiki_filtered.append(word)

    return wiki_filtered


def get_religion_words(
    tokenizer: BertTokenizer, args: CustomArguments
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    with open(file=f"../../data/target/religion/christian.json", mode="r") as christian_fp:
        christian_words = json.load(christian_fp)
    with open(file=f"../../data/target/religion/jewish.json", mode="r") as jewish_fp:
        jewish_words = json.load(jewish_fp)
    with open(file=f"../../data/target/religion/muslim.json", mode="r") as muslim_fp:
        muslim_words = json.load(muslim_fp)

    with open(file=f"../../data/stereotype/stereotype_words.json", mode="r") as stereo_fp:
        stereo_words = json.load(stereo_fp)

    with open(file=f"../../data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        wiki_words = json.load(wiki_fp)
    wiki_words = filter_wiki_words(
        tokenizer=tokenizer,
        wiki_words=wiki_words,
        target_words=christian_words + jewish_words + muslim_words,
        stereo_words=stereo_words,
    )
    wiki_words = wiki_words[: args.num_wiki_words]

    return christian_words, jewish_words, muslim_words, stereo_words, wiki_words


def prepare_stereo_sents(wiki_words: List[str], stereo_words: List[str]) -> List[str]:
    sents = []
    for wiki_word in wiki_words:
        for stereo_word in stereo_words:
            sents.append("[MASK] " + wiki_word + " " + stereo_word + " .")

    return sents


def prepare_neutral_sents(wiki_words: List[str]) -> List[str]:
    sents = []
    for wiki_word_0 in wiki_words:
        for wiki_word_1 in wiki_words:
            sents.append("[MASK] " + wiki_word_0 + " " + wiki_word_1 + " .")

    return sents


def read_data(
    tokenizer: BertTokenizer,
    args: CustomArguments,
) -> Tuple[List[int], List[int], List[int], List[int], List[Dict[str, torch.Tensor]]]:
    dataset = []

    christian_words, jewish_words, muslim_words, stereo_words, wiki_words = get_religion_words(
        tokenizer=tokenizer, args=args
    )

    christian_words, jewish_words, muslim_words = filter_target_words(
        tokenizer=tokenizer, christian_words=christian_words, jewish_words=jewish_words, muslim_words=muslim_words
    )

    christian_ids = tokenizer.convert_tokens_to_ids(christian_words)
    jewish_ids = tokenizer.convert_tokens_to_ids(jewish_words)
    muslim_ids = tokenizer.convert_tokens_to_ids(muslim_words)
    neutral_ids = get_neutral_ids(
        tokenizer=tokenizer,
        christian_words=christian_words,
        jewish_words=jewish_words,
        muslim_words=muslim_words,
        stereo_words=stereo_words,
    )

    stereo_sents = prepare_stereo_sents(wiki_words=wiki_words, stereo_words=stereo_words)
    neutral_sents = prepare_neutral_sents(wiki_words)
    if len(neutral_sents) >= len(stereo_sents):
        neutral_sents = np.random.choice(neutral_sents, size=len(stereo_sents), replace=False).tolist()
    else:
        stereo_sents = np.random.choice(stereo_sents, size=len(neutral_sents), replace=False).tolist()

    for stereo_sent, neutral_sent in zip(stereo_sents, neutral_sents):
        dataset.append({"stereo": stereo_sent, "neutral": neutral_sent})

    return christian_ids, jewish_ids, muslim_ids, neutral_ids, dataset


def collate_fn(tokenizer: BertTokenizer):
    def _collate_fn(data: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        batch = {}
        _batch = {}
        for k in data[0].keys():
            _batch[k] = [datum[k] for datum in data]

        stereo_inputs = tokenizer.__call__(text=_batch["stereo"], padding=True, truncation=True, return_tensors="pt")
        neutral_inputs = tokenizer.__call__(text=_batch["neutral"], padding=True, truncation=True, return_tensors="pt")

        for k in stereo_inputs.keys():
            batch[f"stereo_{k}"] = stereo_inputs[k]
            batch[f"neutral_{k}"] = neutral_inputs[k]
            batch["mask_pos"] = torch.where(stereo_inputs["input_ids"] == tokenizer.mask_token_id)[-1]

        return batch

    return _collate_fn


def prepare_dataloader(dataset: CustomDataset, tokenizer: BertTokenizer, args: CustomArguments) -> DataLoader:
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn(tokenizer),
        pin_memory=True,
    )

    return train_dataloader
