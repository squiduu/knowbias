import json
from typing import Dict, List, Tuple

from config import MyArguments
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers.models.bert.tokenization_bert import BertTokenizer


class CustumDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


def get_gender_words(
    args: MyArguments,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    with open(
        file=f"../data/target/gender/male/male_words_{args.num_target_words}.json",
        mode="r",
    ) as m_fp:
        M_WORDS = json.load(m_fp)
    M_WORDS = M_WORDS[: args.num_target_words]
    with open(
        file=f"../data/target/gender/female/female_words_{args.num_target_words}.json",
        mode="r",
    ) as f_fp:
        F_WORDS = json.load(f_fp)
    F_WORDS = F_WORDS[: args.num_target_words]

    with open(file=f"../data/stereotype/stereotype_words.json", mode="r") as ster_fp:
        STER_WORDS = json.load(ster_fp)

    with open(file=f"../data/wiki/wiki_words_5000.json", mode="r") as wiki_fp:
        WIKI_WORDS = json.load(wiki_fp)
    WIKI_WORDS = filter_wiki(
        wiki_words=WIKI_WORDS, targ_words=M_WORDS + F_WORDS, ster_words=STER_WORDS
    )
    WIKI_WORDS = WIKI_WORDS[: args.num_wiki_words]

    return M_WORDS, F_WORDS, STER_WORDS, WIKI_WORDS


def filter_wiki(wiki_words: List[str], targ_words: List[str], ster_words: List[str]):
    filtered = []
    for word in wiki_words:
        if word not in (targ_words + ster_words):
            filtered.append(word)

    return filtered


def prepare_stereo_sents(
    targ_words: List[str], wiki_words: List[str], ster_words: List[str]
) -> List[str]:
    sents = []
    for targ_word in targ_words:
        for wiki_word in wiki_words:
            for ster_word in ster_words:
                sents.append(targ_word + " " + wiki_word + " " + ster_word + " .")

    return sents


def prepare_neutral_sents(targ_words: List[str], wiki_words: List[str]) -> List[str]:
    sents = []
    for targ_word in targ_words:
        for wiki_word in wiki_words:
            for wiki_word in wiki_words:
                sents.append(targ_word + " " + wiki_word + " " + wiki_word + " .")

    return sents


def read_data(args: MyArguments) -> List[Dict[str, str]]:
    #
    dataset = []

    #
    M_WORDS, F_WORDS, STER_WORDS, WIKI_WORDS = get_gender_words(args)

    #
    m_sents = prepare_stereo_sents(
        targ_words=M_WORDS,
        wiki_words=WIKI_WORDS[: args.num_stereo_words],
        ster_words=STER_WORDS,
    )
    f_sents = prepare_stereo_sents(
        targ_words=F_WORDS,
        wiki_words=WIKI_WORDS[: args.num_stereo_words],
        ster_words=STER_WORDS,
    )
    #
    n_sents = prepare_neutral_sents(targ_words=M_WORDS + F_WORDS, wiki_words=WIKI_WORDS)
    n_sents = n_sents[: len(m_sents)]

    #
    for m_sent, f_sent, n_sent in zip(m_sents, f_sents, n_sents):
        dataset.append({"m": m_sent, "f": f_sent, "n": n_sent})

    return dataset


def collate_fn(tokenizer: BertTokenizer):
    def _collate(data):
        batch = {}

        for k in data[0]:
            batch[k] = [datum[k] for datum in data]

        m_batch = tokenizer.__call__(
            text=batch["m"], padding=True, return_tensors="pt", truncation=True
        )
        f_batch = tokenizer.__call__(
            text=batch["f"], padding=True, return_tensors="pt", truncation=True
        )
        n_batch = tokenizer.__call__(
            text=batch["n"], padding=True, return_tensors="pt", truncation=True
        )

        for k in m_batch.keys():
            batch[f"m_{k}"] = m_batch[k]
            batch[f"f_{k}"] = f_batch[k]
            batch[f"n_{k}"] = n_batch[k]

        return batch

    return _collate


def prepare_dataloader(tokenizer: BertTokenizer, args: MyArguments):
    #
    train_ds = CustumDataset(read_data(args))

    #
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn(tokenizer),
        pin_memory=True,
    )

    return train_dl
