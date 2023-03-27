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


FIXED_PARAMS = dict(
    # fixed parameters
    index=0,
    blur=0,
    width=-1,
    skewing_angle=0,
    orientation=0,
    name_format=0,
    output_bboxes=0,
    distorsion_type=0,
    background_type=0,
    distorsion_orientation=0,
    alignment=0,
    space_width=1,
    character_spacing=0,
    out_dir=None,
    extension=None,
    word_split=False,
    random_skew=False,
    output_mask=False,
    is_handwritten=False,
    random_blur=False,
    image_mode="RGB",
    image_dir="",
)

def generate_line(text):
    rand_params = dict(
        # random parameters
        font=random.choice(FONTS),
        text_color=random.choice(COLORS),
        stroke_fill=random.choice(COLORS),
        stroke_width=random.randint(0, 3),
        size=random.randint(64, 128),
        margins=[random.randint(0, 10)]*4,
        fit=random.choice([True, False]),
    )

    try:
        image = FakeTextDataGenerator.generate(
            text=text,
            **FIXED_PARAMS,
            **rand_params,
        )
        if image is None:
            raise Exception("image is None")
    
    except Exception as e:
        # sometimes the generator fails due to a bad combination of parameters
        image, rand_params = generate_line(text)

    rand_params['font'] = rand_params['font'].split('/')[-1]

    return image, rand_params
