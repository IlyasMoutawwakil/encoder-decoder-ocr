import json
from pathlib import Path

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        characters,
        model_max_length,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        unk_token_id=3,
        *args,
        **kwargs
    ):
        """Character tokenizer for Hugging Face transformers."""

        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[EOS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            *args,
            **kwargs,
        )

        self._vocab_str_to_int = {
            "[BOS]": bos_token_id,
            "[EOS]": eos_token_id,
            "[PAD]": pad_token_id,
            "[UNK]": unk_token_id,
            **{ch: i for i, ch in enumerate(characters, start=unk_token_id + 1)},
        }
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()
        }

    @property
    def vocab_size(self):
        return len(self._vocab_str_to_int)

    def _tokenize(self, text):
        return list(text)

    def _convert_token_to_id(self, token):
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index):
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0, token_ids_1=None
    ):
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        result = bos + token_ids_0 + eos

        return result

    def get_special_tokens_mask(
        self,
        token_ids_0,
        token_ids_1=None,
        already_has_special_tokens=False,
    ):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]

        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0, token_ids_1=None
    ):
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]

        result = len(bos + token_ids_0 + eos) * [0]

        return result

    def get_config(self):
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config):
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory, **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory, **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)


if __name__ == '__main__':

    # tokenizer config
    characters = list("abcdefghijklmnopqrstuvwxyz0123456789")
    model_max_length = 10

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=characters,
        model_max_length=model_max_length,
    )

    texts = ["a9c", "b9c", "c9c"]

    # test encode
    encoded = tokenizer(
        texts, return_tensors="pt", padding="max_length")
    print(encoded)

    # test decode
    decoded = tokenizer.batch_decode(
        encoded['input_ids'], skip_special_tokens=True)
    print(decoded)

    # test save and load
    tokenizer.save_pretrained("./")
    tokenizer = CharacterTokenizer.from_pretrained("./")
