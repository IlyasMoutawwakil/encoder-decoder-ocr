import random

NUMBERS = """0123456789"""
ALPHABET = """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""

SPECIAL_ALPHABET = """ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ"""
SPECIAL_CHARCTERS = """!"#$£€%§&½\'°()*+,-./:;<=>?@[\\]^_`{|}~“”‘’«» """


def clean_df(df):

    # remove characters that are not numbers, alphabet, special alphabet or special characters
    df['text'] = df['text'].str.replace(
        f"[^{NUMBERS}{ALPHABET}{SPECIAL_ALPHABET}{SPECIAL_CHARCTERS}]", '', regex=True)

    # remove trailing spaces
    df['text'] = df['text'].str.strip()

    # remove multiple spaces
    df['text'] = df['text'].str.replace('\s+', ' ', regex=True)

    # remove lines with only spaces
    df = df[df['text'].str.replace(' ', '', regex=False).str.len() > 0]

    # remove empty lines
    df = df[df['text'].str.len() > 0]

    # remove repeated lines
    df = df.drop_duplicates(subset=['text'], keep='first')

    return df

def get_random_cut(phrase, max_length=94):
    
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