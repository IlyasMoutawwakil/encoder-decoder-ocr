from trdg.data_generator import FakeTextDataGenerator
from trdg.utils import load_fonts
from dataset.utils import get_random_cut

from torchvision.datasets import VisionDataset
import numpy as np
import random
import torch
import os

FONTS = load_fonts('fr')

class WikiLinesDataset(VisionDataset):
    """
    A custom dataset to load wikipedia text and generate images from it
    """

    def __init__(
            self,
            paragraphs,
            tokenizer,
    ):
        self.paragraphs = paragraphs
        self.tokenizer = tokenizer
        
        self.counter = 0

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):

        text = get_random_cut(
            self.paragraphs[idx],
            max_length=self.tokenizer.model_max_length - 2, # -2 for the [BOS] and [EOS] tokens
        ) 
        
        image, _ = self.generate_image(text)
        tokens = self.tokenizer(
            text,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        features = torch.tensor(np.array(image)).permute(2, 0, 1) / 127.5 - 1
        target = tokens["input_ids"].squeeze(0)
        mask = tokens["attention_mask"].squeeze(0)

        return {
            "features": features,
            "target": target,
            "mask": mask,
        }

    def generate_image(self, text):

        fixed_params = dict(
            width=-1,
            blur=1,
            skewing_angle=2,
            orientation=0,
            name_format=0,
            output_bboxes=0,
            stroke_width = 0,
            alignment=0,
            out_dir=None,
            extension=None,
            word_split=False,
            output_mask=False,
            is_handwritten=False,
            image_dir=os.path.join("..", os.path.split(os.path.realpath(__file__))[0], "images"),
            text_color="#282828",
            stroke_fill="#282828",
            image_mode="RGB",
        )

        random_params = dict(
            font=random.choice(FONTS),
            size= random.randint(64, 128),
            background_type=random.randint(0, 2), # 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures
            distorsion_type=random.randint(0, 4), # 0: None (Default), 1: Sine wave, 2: Cosine wave, 3:Random
            distorsion_orientation=random.randint(0, 3), # 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both
            random_skew=random.choice([True, False]),
            random_blur=random.choice([True, False]),
            space_width=random.randint(1, 3), # multiplied by the normal space width
            character_spacing=random.randint(1, 10), # number of pixels
            margins=(random.randint(0, 10) for _ in range(4)), # in pixels
            fit=random.choice([True, False]),
        )

        image =  FakeTextDataGenerator.generate(
            text=text,
            index=self.counter,
            **random_params,
            **fixed_params
        )
        
        self.counter += 1
        
        return image, random_params


if __name__ == '__main__':
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt

    # get a generic tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-cased", 
        model_max_length=100
    )

    # create a dataset
    dataset = WikiLinesDataset(
        paragraphs=[
            "Hello world",
            ' '.join(["Hello world"]*20),
        ],
        tokeniser=tokenizer,
    )

    # get a sample from the dataset
    sample = dataset[0]

    # print the shapes and dtypes of the sample
    print(sample["features"].shape, sample["features"].dtype)
    print(sample["target"].shape, sample["target"].dtype)
    print(sample["mask"].shape, sample["mask"].dtype)

    # decode the target
    print(tokenizer.decode(sample["target"], skip_special_tokens=True))

    # show the image

    for i in range(10):
        # get a sample from the dataset
        sample = dataset[i % 2]
        plt.imshow((sample["features"].permute(1, 2, 0) + 1) / 2)
        plt.show()
        