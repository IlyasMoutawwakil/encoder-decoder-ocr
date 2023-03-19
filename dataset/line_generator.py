import random
from trdg.utils import load_fonts
from trdg.data_generator import FakeTextDataGenerator


BAD_FONTS = [
    "Capture_it_2.ttf",
    "BEBAS___.ttf",
    "SEASRN__.ttf",
    "Capture_it.ttf",
    "Walkway_Oblique_SemiBold.ttf",
]
FONTS = [font for font in load_fonts('fr') if font not in BAD_FONTS]

COLORS = [
    "#000000",
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#00FFFF",
    "#FF00FF",
]


def generate_line(text):
    gen_params = dict(
        # fixed parameters
        index=0,
        blur=1,
        width=-1,
        skewing_angle=1,
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
        
        # random parameters
        font=random.choice(FONTS),
        text_color=random.choice(COLORS),
        stroke_fill=random.choice(COLORS),
        stroke_width=random.randint(0, 3),  # stroke width in pixels
        size=random.randint(64, 128),  # height in pixels
        # 0: Gaussian Noise, 1: Plain white
        background_type=random.choice([0, 1]),
        # 0: None (Default), 1: Sine wave, 2: Cosine wave, 3:Random
        distorsion_type=random.choice([0, 3]),
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
        margins=[random.randint(0, 10)]*4,
        # if true, tight crop the image to the text
        fit=random.choice([True, False]),
    )

    try:
        image = FakeTextDataGenerator.generate(
            text=text,
            **gen_params,
        )
    except Exception as e:
        # sometimes the generator fails due to a bad combination of parameters
        return generate_line(text)

    return image
