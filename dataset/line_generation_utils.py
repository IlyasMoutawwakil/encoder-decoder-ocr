import random
from trdg.utils import load_fonts
from trdg.data_generator import FakeTextDataGenerator

FONTS = load_fonts('fr')
COLORS = [
    "#000000",
    # "#FFFFFF", # white is not a good color for text
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#00FFFF",
    "#FF00FF",
]

FIXED_PARAMS = dict(
    index=0,
    blur=1,
    width=-1,
    skewing_angle=2,
    orientation=0,
    name_format=0,
    output_bboxes=0,

    alignment=0,
    out_dir=None,
    extension=None,
    word_split=False,
    output_mask=False,
    is_handwritten=False,
    image_mode="RGB",
    image_dir="",
)


def generate_line(text):
    random_params = dict(
        font=random.choice(FONTS),
        text_color=random.choice(COLORS),
        stroke_fill=random.choice(COLORS),
        stroke_width=random.randint(0, 3),  # stroke width in pixels
        size=random.randint(64, 128),  # height in pixels
        # 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures
        # 3 is not interesting for us
        background_type=random.choice([0, 1, 2]),
        # 0: None (Default), 1: Sine wave, 2: Cosine wave, 3:Random
        distorsion_type=random.choice([0, 1, 2, 3]),
        # 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both
        distorsion_orientation=random.choice([0, 1, 2]),
        # if true, the skewing angle is random between -skewing_angle and +skewing_angle
        random_skew=random.choice([True, False]),
        # if true, the blur is random between 0 and blur
        random_blur=random.choice([True, False]),
        # multiplied by normal words level spacing
        space_width=random.randint(1, 3),
        # characters level spacing in pixels
        character_spacing=random.randint(1, 10),
        # margins in pixels
        margins=(random.randint(0, 10) for _ in range(4)), 
        # if true, tight crop the image to the text
        fit=random.choice([True, False]),
    )

    image = FakeTextDataGenerator.generate(
        text=text,
        **random_params,
        **FIXED_PARAMS
    )

    return image
