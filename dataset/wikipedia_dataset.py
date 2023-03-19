try:
    from dataset.textline_dataset import TextLineDataset
except ImportError:
    from textline_dataset import TextLineDataset

from torch.utils.data import DataLoader
from functools import partial
import datasets as ds
import lightning as L
import torch
import re

NUMBERS = "0123456789"
SPECIAL_CHARCTERS = """!"#$£€%§&½'°()*+,-./:;<=>?@[\]^_`{|}~“”‘’«» """

LATIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz"
EXTRA_LATIN_ALPHABET = "ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ" + "àâæçéèêëîïôœùûüÿ"

FRENCH_CHARACTERS = NUMBERS + SPECIAL_CHARCTERS + \
    LATIN_ALPHABET + EXTRA_LATIN_ALPHABET


class WikipediaTextLineDataModule(L.LightningDataModule):
    def __init__(self, name, transform, tokenizer, batch_size, num_workers, characters):
        super().__init__()

        self.name = name
        self.transform = transform
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.characters = characters
        self.collate_fn = partial(
            tokenization_collate_fn, tokenizer=self.tokenizer)

    def prepare_data(self):
        ds.load_dataset("wikipedia", self.name, split="train[:90%]")
        ds.load_dataset("wikipedia", self.name, split="train[-10%:]")

    def setup(self, stage=None):
        train_dataset = ds.load_dataset(
            "wikipedia", self.name, split="train[:90%]")
        val_dataset = ds.load_dataset(
            "wikipedia", self.name, split="train[-10%:]")

        train_dataset = preprocess_dataset(
            train_dataset, text_column='text',
            num_proc=self.num_workers, characters=self.characters
        )
        val_dataset = preprocess_dataset(
            val_dataset, text_column='text',
            num_proc=self.num_workers, characters=self.characters
        )

        self.train_dataset = TextLineDataset(
            dataset=train_dataset,
            transform=self.transform,
            model_max_length=self.tokenizer.model_max_length,
        )

        self.val_dataset = TextLineDataset(
            dataset=val_dataset,
            transform=self.transform,
            model_max_length=self.tokenizer.model_max_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


def preprocess_dataset(dataset, text_column, num_proc, characters):
    dataset = dataset.map(
        lambda x: {
            'lines': preprocess_paragraph(x[text_column], characters=characters)
        },
        remove_columns=[text_column],
        num_proc=num_proc,
    )

    return dataset


def preprocess_paragraph(text, characters):
    lines = []
    for line in text.split('\n'):
        line = preprocess_line(line, characters=characters)
        if len(line) > 0:
            lines.append(line)
    return lines


def preprocess_line(text, characters):
    # does the same as clean_df but for a single text
    text = re.sub(f"[^{characters}]", '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def tokenization_collate_fn(batch, tokenizer):
    pixels = torch.stack([sample["pixels"] for sample in batch])

    tokens = tokenizer(
        [sample["text"] for sample in batch],
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        padding="longest",
        return_tensors="pt",
    )['input_ids']

    return {
        "pixels": pixels,
        "tokens": tokens,
    }
