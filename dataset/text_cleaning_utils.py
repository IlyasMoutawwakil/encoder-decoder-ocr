import re
import random

NUMBERS = """0123456789"""
ALPHABET = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""

SPECIAL_ALPHABET = """ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ"""
SPECIAL_CHARCTERS = """!"#$£€%§&½'°()*+,-./:;<=>?@[\]^_`{|}~“”‘’«» """

ALL_CHARACTERS = NUMBERS + ALPHABET + SPECIAL_ALPHABET + SPECIAL_CHARCTERS


def clean_line(text):
    # does the same as clean_df but for a single text
    text = re.sub(f"[^{ALL_CHARACTERS}]", '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_wikipedia_dataset(dataset):
    dataset = dataset.map(
        lambda x: {'lines': x['text'].split('\n')},
        remove_columns=['text'],
        num_proc=4,
    )
    dataset = dataset.map(
        lambda x: {'lines': [clean_line(line) for line in x['lines']]},
        remove_columns=['lines'],
        num_proc=4,
    )
    dataset = dataset.map(
        lambda x: {'lines': [line for line in x['lines'] if len(line) > 0]},
        remove_columns=['lines'],
        num_proc=4,
    )

    return dataset


def get_random_cut(text, max_length):
    if len(text) <= max_length:
        return text

    else:
        words = text.split(' ')
        first_word_index = random.randint(0, len(words) - 1)
        new_text = words[first_word_index]

        for i in range(first_word_index + 1, len(words)):
            candidate = new_text + ' ' + words[i]
            if len(candidate) <= max_length:
                new_text = candidate
            else:
                break

        return new_text
