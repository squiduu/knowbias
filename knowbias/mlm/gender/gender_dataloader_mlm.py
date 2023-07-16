import json
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from config import GuidebiasArguments
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer


class GuidebiasDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


def filter_target_words(
    tokenizer: BertTokenizer, male_words: List[str], female_words: List[str]
) -> Tuple[List[str], List[str]]:
    male_filtered = []
    female_filtered = []
    for i in range(max(len(male_words), len(female_words))):
        if tokenizer.unk_token_id not in tokenizer.convert_tokens_to_ids([male_words[i], female_words[i]]):
            male_filtered.append(male_words[i])
            female_filtered.append(female_words[i])

    return male_filtered, female_filtered


def get_neutral_ids(
    tokenizer: BertTokenizer, male_words: List[str], female_words: List[str], stereo_words: List[str]
) -> List[int]:
    male_tokens = []
    for word in male_words:
        male_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    female_tokens = []
    for word in female_words:
        female_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))
    stereo_tokens = []
    for word in stereo_words:
        stereo_tokens.extend(tokenizer.convert_ids_to_tokens(tokenizer.encode(word, add_special_tokens=False)))

    neutral_ids = [
        id
        for token, id in tokenizer.get_vocab().items()
        if token not in list(set(stereo_tokens)) + list(set(male_tokens)) + list(set(female_tokens))
    ]

    return neutral_ids


def filter_wiki_words(
    tokenizer: BertTokenizer, wiki_words: List[str], target_words: List[str], stereo_words: List[str]
) -> List[str]:
    filtered = []
    for word in wiki_words:
        if (
            word not in (target_words + stereo_words)
            and tokenizer.convert_tokens_to_ids(word) != tokenizer.unk_token_id
        ):
            filtered.append(word)

    return filtered


def get_gender_words(
    tokenizer: BertTokenizer, args: GuidebiasArguments
) -> Tuple[List[str], List[str], List[str], List[str]]:
    with open(file=f"../../../data/target/gender/male/male_words_{args.num_target_words}.json", mode="r") as male_fp:
        male_words = json.load(male_fp)
    with open(
        file=f"../../../data/target/gender/female/female_words_{args.num_target_words}.json", mode="r"
    ) as female_fp:
        female_words = json.load(female_fp)

    with open(file=f"../../../data/stereotype/stereotype_words.json", mode="r") as stereo_fp:
        stereo_words = json.load(stereo_fp)

    with open(file=f"../../../data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        wiki_words = json.load(wiki_fp)
    wiki_words = filter_wiki_words(
        tokenizer=tokenizer, wiki_words=wiki_words, target_words=male_words + female_words, stereo_words=stereo_words
    )

    return male_words, female_words, stereo_words, wiki_words


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
    tokenizer: BertTokenizer, args: GuidebiasArguments
) -> Tuple[List[str], Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]]]:
    dataset = []

    male_words, female_words, stereo_words, wiki_words = get_gender_words(tokenizer=tokenizer, args=args)

    neutral_ids = get_neutral_ids(
        tokenizer=tokenizer, male_words=male_words, female_words=female_words, stereo_words=stereo_words
    )
    male_words, female_words = filter_target_words(
        tokenizer=tokenizer, male_words=male_words, female_words=female_words
    )
    male_ids = tokenizer.convert_tokens_to_ids(male_words)
    female_ids = tokenizer.convert_tokens_to_ids(female_words)

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

    return male_ids, female_ids, neutral_ids, dataset


def prepare_dataloader(dataset: GuidebiasDataset, args: GuidebiasArguments) -> DataLoader:
    train_dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    return train_dataloader
