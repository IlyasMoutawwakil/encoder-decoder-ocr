try:
    from dataset.textline_dataset import TextLineDataset
except ImportError:
    from textline_dataset import TextLineDataset

from torch.utils.data import DataLoader
from functools import partial
import datasets as ds
import pytorch_lightning as pl
import torch
import re


class WikipediaTextLineDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, tokenizer, batch_size, num_workers, allowed_characters, train_transform=None, val_transform=None):
        super().__init__()

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.allowed_characters = allowed_characters
        self.collate_fn = partial(
            tokenization_collate_fn,
            tokenizer=self.tokenizer
        )
        
        self.train_transform = train_transform
        self.val_transform = val_transform

    def prepare_data(self):
        ds.load_dataset("wikipedia", self.dataset_name, split="train[:10%]")
        ds.load_dataset("wikipedia", self.dataset_name, split="train[-10%:]")

    def setup(self, stage=None):
        train_dataset = ds.load_dataset(
            "wikipedia", self.dataset_name, split="train[:10%]")
        val_dataset = ds.load_dataset(
            "wikipedia", self.dataset_name, split="train[-10%:]")

        train_dataset = preprocess_dataset(
            train_dataset,
            num_proc=self.num_workers, 
            allowed_characters=self.allowed_characters
        )
        val_dataset = preprocess_dataset(
            val_dataset,
            num_proc=self.num_workers, 
            allowed_characters=self.allowed_characters
        )

        self.train_dataset = TextLineDataset(
            dataset=train_dataset,
            transform=self.train_transform,
            model_max_length=self.tokenizer.model_max_length,
        )

        self.val_dataset = TextLineDataset(
            dataset=val_dataset,
            transform=self.val_transform,
            model_max_length=self.tokenizer.model_max_length,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2 * (self.batch_size // self.num_workers),
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=2 * (self.batch_size // self.num_workers),
            pin_memory=True,
            shuffle=False,
        )


def preprocess_dataset(dataset, num_proc, allowed_characters):
    dataset = dataset.map(
        lambda x: {
            'lines': preprocess_paragraph(x['text'], allowed_characters=allowed_characters)
        },
        remove_columns=['text'],
        num_proc=num_proc,
    )

    return dataset


def preprocess_paragraph(text, allowed_characters):
    lines = []
    for line in text.split('\n'):
        line = preprocess_line(line, allowed_characters=allowed_characters)
        if len(line) > 0:
            lines.append(line)
    return lines


def preprocess_line(text, allowed_characters):
    # does the same as clean_df but for a single text
    text = re.sub(f"[^{allowed_characters}]", '', text)
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
