import re
import random

NUMBERS = """0123456789"""
ALPHABET = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""

SPECIAL_ALPHABET = """ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ"""
SPECIAL_CHARCTERS = """!"#$£€%§&½\'°()*+,-./:;<=>?@[\\]^_`{|}~“”‘’«» """

ALL_CHARACTERS = NUMBERS + ALPHABET + SPECIAL_ALPHABET + SPECIAL_CHARCTERS


def clean_text(text):
    # does the same as clean_df but for a single text
    text = re.sub(f"[^{ALL_CHARACTERS}]", '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def clean_hf_dataset(dataset):
    dataset = dataset.map(
        lambda x: {'paragraph': clean_text(x['paragraph'])},
        remove_columns=['paragraph']
    )
    dataset = dataset.filter(
        lambda x: len(re.sub('\s', '', x['paragraph'])) > 0
    )

    return dataset


def get_random_cut(phrase, max_length):
    if len(phrase) <= max_length:
        return phrase
    
    else:
        words = phrase.split(' ')
        first_word_index = random.randint(0, len(words) - 1)
        new_phrase = words[first_word_index]

        for i in range(first_word_index + 1, len(words)):
            candidate = new_phrase + ' ' + words[i]
            if len(candidate) <= max_length:
                new_phrase = candidate
            else:
                break

        return new_phrase
