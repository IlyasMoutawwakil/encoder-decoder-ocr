from modeling.encoder import SwinTransformerEncoder
from modeling.decoder import AutoregressiveTransformerDecoder

from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler

NUMBERS = "0123456789"
SPECIAL_CHARCTERS = """!"#$£€%§&½'°()*+,-./:;<=>?@[\]^_`{|}~“”‘’«» """

LATIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz"
EXTRA_LATIN_ALPHABET = "ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ" + "àâæçéèêëîïôœùûüÿ"

CHARACTERS = {
    "fr": NUMBERS + SPECIAL_CHARCTERS + LATIN_ALPHABET + EXTRA_LATIN_ALPHABET,
    "en": NUMBERS + SPECIAL_CHARCTERS + LATIN_ALPHABET,
}

DATASET_NAME = {
    "fr": "20220301.fr",
    "en": "20220301.en",
}


ENCODER_CONFIG = {
    'SwinTransformerEncoder': {
        'class': SwinTransformerEncoder,
        'params': dict(
            height=128,
            width=640,
            channels=1,
            patch_size=4,
            window_size=8,
            embed_dim=96,
            depths=[2, 6, 2],
            num_heads=[6, 12, 24],
        )
    },
}

DECODER_CONFIG = {
    'AutoregressiveTransformerDecoder': {
        'class': AutoregressiveTransformerDecoder,
        'params': dict(
            dim=384,
            heads=8,
            dropout=0.1,
            activation='gelu',
            norm_first=True,
            num_layers=4,
        )
    },
}

OPTIMIZER_CONFIG = {
    'AdamW': {
        'class': AdamW,
        'params': dict(
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
        )
    },
}

# scheduler config
SCHEDULER_CONFIG = {
    'CosineLRScheduler': {
        'class': CosineLRScheduler,
        'params': dict(
            t_initial=200,
            lr_min=1e-6,
            cycle_mul=3,
            cycle_decay=0.8,
            cycle_limit=20,
            warmup_t=20,
            k_decay=1.5,
        )
    },
}
