"""Read the existing classification splits; in scratch mode the vocabulary is built from the training set only."""

import csv
import re
from collections import Counter
from pathlib import Path

import torch


DATASETS = ("mr", "sst2")
SPLITS = ("train", "validation", "test")
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\wxc\Desktop\xuexi\eassy\PHD\AAAI\Fanhua_duibi\SentiCSE-main\dataset"
)
TOKEN_PATTERN = re.compile(r"[\w]+(?:['’][\w]+)*|[^\w\s]", flags=re.UNICODE)
SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")


def read_splits(root, dataset, split_dir):
    splits = {}
    for split in SPLITS:
        path = Path(root) / dataset / split_dir / f"{split}.csv"
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            if not {"text", "label"}.issubset(rows.fieldnames or []):
                raise ValueError(f"{path} must contain text and label columns")
            items = []
            for row in rows:
                text, label = row["text"], (row["label"] or "").strip()
                if not text or not text.strip() or label not in {"0", "1"}:
                    raise ValueError(f"Invalid text or 0/1 label at {path}:{rows.line_num}")
                items.append((text, int(label)))
            if not items:
                raise ValueError(f"Empty split: {path}")
            splits[split] = items
    return splits


def tokens(text):
    return TOKEN_PATTERN.findall(text.lower())


def build_vocab(rows, max_vocab, min_freq):
    counts = Counter(token for text, _ in rows for token in tokens(text))
    words = sorted((word for word, count in counts.items() if count >= min_freq),
                   key=lambda word: (-counts[word], word))
    return {word: i for i, word in enumerate((*SPECIAL_TOKENS, *words[:max_vocab - 4]))}


class BatchCollator:
    def __init__(self, max_length, *, vocab=None, tokenizer=None):
        self.max_length = max_length
        self.vocab = vocab
        self.tokenizer = tokenizer

    def encode(self, texts):
        if self.tokenizer is not None:
            batch = self.tokenizer(list(texts), max_length=self.max_length, truncation=True,
                                   padding=True, return_tensors="pt")
            return batch["input_ids"], batch["attention_mask"].bool()
        sequences = [[2, *(self.vocab.get(w, 1) for w in tokens(text)[:self.max_length - 2]), 3]
                     for text in texts]
        width = max(map(len, sequences))
        ids = torch.tensor([seq + [0] * (width - len(seq)) for seq in sequences], dtype=torch.long)
        return ids, ids.ne(0)

    def __call__(self, rows):
        texts, labels = zip(*rows)
        ids, mask = self.encode(texts)
        return ids, mask, torch.tensor(labels, dtype=torch.long)
