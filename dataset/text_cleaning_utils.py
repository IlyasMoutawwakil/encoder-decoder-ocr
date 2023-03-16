import re
import random

NUMBERS = "0123456789"
SPECIAL_CHARCTERS = """!"#$£€%§&½'°()*+,-./:;<=>?@[\]^_`{|}~“”‘’«» """

LATIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "abcdefghijklmnopqrstuvwxyz"
EXTRA_LATIN_ALPHABET = "ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ" + "àâæçéèêëîïôœùûüÿ"

FRENCH_CHARACTERS = NUMBERS + SPECIAL_CHARCTERS + \
    LATIN_ALPHABET + EXTRA_LATIN_ALPHABET


def preprocess_line(text):
    # does the same as clean_df but for a single text
    text = re.sub(f"[^{FRENCH_CHARACTERS}]", '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    return text


def preprocess_paragraph(text):
    lines = []
    for line in text.split('\n'):
        line = preprocess_line(line)
        if len(line) > 0:
            lines.append(line)
    return lines


def preprocess_wikipedia_dataset(dataset, text_column='text', num_proc=4):
    dataset = dataset.map(
        lambda x: {
            'lines': preprocess_paragraph(x[text_column])
        },
        remove_columns=[text_column],
        num_proc=num_proc,
    )

    return dataset


def get_random_cut(text, max_length=96):
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
