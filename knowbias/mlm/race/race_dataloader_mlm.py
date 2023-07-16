import json
from typing import Dict, List, Tuple, Union

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


def filter_target_words(
    tokenizer: BertTokenizer, asian_words: List[str], black_words: List[str], caucasian_words: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    asian_filtered = []
    black_filtered = []
    caucasian_filtered = []
    for i in range(max(len(asian_words), len(black_words), len(caucasian_words))):
        if tokenizer.unk_token_id not in tokenizer.convert_tokens_to_ids(
            [asian_words[i], black_words[i], caucasian_words[i]]
        ):
            asian_filtered.append(asian_words[i])
            black_filtered.append(black_words[i])
            caucasian_filtered.append(caucasian_words[i])

    return asian_filtered, black_filtered, caucasian_filtered


def get_neutral_ids(
    tokenizer: BertTokenizer,
    asian_words: List[str],
    black_words: List[str],
    caucasian_words: List[str],
    stereo_words: List[str],
) -> List[int]:
    asian_tokens = []
    black_tokens = []
    caucasian_tokens = []
    stereo_tokens = []
    for word in asian_words:
        asian_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    for word in black_words:
        black_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    for word in caucasian_words:
        caucasian_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    for word in stereo_words:
        stereo_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))

    neutral_ids = [
        id
        for token, id in tokenizer.get_vocab().items()
        if token
        not in list(set(asian_tokens))
        + list(set(black_tokens))
        + list(set(caucasian_tokens))
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


def get_race_words(tokenizer: BertTokenizer) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    with open(file=f"../../../data/target/race/asian.json", mode="r") as asian_fp:
        asian_words = json.load(asian_fp)
    with open(file=f"../../../data/target/race/black.json", mode="r") as black_fp:
        black_words = json.load(black_fp)
    with open(file=f"../../../data/target/race/caucasian.json", mode="r") as caucasian_fp:
        caucasian_words = json.load(caucasian_fp)

    with open(file=f"../../../data/stereotype/stereotype_words.json", mode="r") as stereo_fp:
        stereo_words = json.load(stereo_fp)

    with open(file=f"../../../data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        wiki_words = json.load(wiki_fp)
    wiki_words = filter_wiki_words(
        tokenizer=tokenizer,
        wiki_words=wiki_words,
        target_words=asian_words + black_words + caucasian_words,
        stereo_words=stereo_words,
    )

    return asian_words, black_words, caucasian_words, stereo_words, wiki_words


def prepare_stereo_sentences(wiki_words: List[str], stereo_words: List[str]) -> List[str]:
    sentences = []
    for wiki_word in wiki_words:
        for stereo_word in stereo_words:
            sentences.append("[MASK] " + wiki_word + " " + stereo_word + " .")

    return sentences


def prepare_neutral_sentences(wiki_words: List[str]) -> List[str]:
    sentences = []
    for wiki_word_0 in wiki_words:
        for wiki_word_1 in wiki_words:
            sentences.append("[MASK] " + wiki_word_0 + " " + wiki_word_1 + " .")

    return sentences


def read_data(
    tokenizer: BertTokenizer,
) -> Tuple[List[int], List[int], List[int], List[int], List[Dict[str, torch.Tensor]]]:
    dataset = []

    asian_words, black_words, caucasian_words, stereo_words, wiki_words = get_race_words(tokenizer)

    neutral_ids = get_neutral_ids(
        tokenizer=tokenizer,
        asian_words=asian_words,
        black_words=black_words,
        caucasian_words=caucasian_words,
        stereo_words=stereo_words,
    )
    asian_words, black_words, caucasian_words = filter_target_words(
        tokenizer=tokenizer, asian_words=asian_words, black_words=black_words, caucasian_words=caucasian_words
    )
    asian_ids = tokenizer.convert_tokens_to_ids(asian_words)
    black_ids = tokenizer.convert_tokens_to_ids(black_words)
    caucasian_ids = tokenizer.convert_tokens_to_ids(caucasian_words)

    stereo_sentences = prepare_stereo_sentences(wiki_words=wiki_words, stereo_words=stereo_words)
    neutral_sentences = prepare_neutral_sentences(wiki_words)
    if len(neutral_sentences) >= len(stereo_sentences):
        neutral_sentences = np.random.choice(neutral_sentences, size=len(stereo_sentences), replace=False).tolist()
    else:
        stereo_sentences = np.random.choice(stereo_sentences, size=len(neutral_sentences), replace=False).tolist()

    stereo_inputs = tokenizer.__call__(text=stereo_sentences, padding=True, truncation=True, return_tensors="pt")
    neutral_inputs = tokenizer.__call__(text=neutral_sentences, padding=True, truncation=True, return_tensors="pt")
    mask_pos = np.where(torch.Tensor.numpy(stereo_inputs["input_ids"]) == tokenizer.mask_token_id)[1]

    for i in range(torch.Tensor.size(stereo_inputs["input_ids"])[0]):
        dataset.append(
            {
                "stereo_input_ids": stereo_inputs["input_ids"][i],
                "stereo_attention_mask": stereo_inputs["attention_mask"][i],
                "stereo_token_type_ids": stereo_inputs["token_type_ids"][i],
                "neutral_input_ids": neutral_inputs["input_ids"][i],
                "neutral_attention_mask": neutral_inputs["attention_mask"][i],
                "neutral_token_type_ids": neutral_inputs["token_type_ids"][i],
                "mask_pos": mask_pos[i],
            }
        )

    return asian_ids, black_ids, caucasian_ids, neutral_ids, dataset


def prepare_dataloader(dataset: GuidebiasDataset, args: GuidebiasArguments) -> DataLoader:
    train_dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    return train_dataloader
