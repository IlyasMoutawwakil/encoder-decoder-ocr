try:
    from dataset.line_generator import generate_line
except ImportError:
    from line_generator import generate_line

from torchvision.datasets import VisionDataset
import random


class TextLineDataset(VisionDataset):
    def __init__(self, dataset, model_max_length, transform=None):
        super().__init__(root=None)

        self.dataset = dataset
        self.model_max_length = model_max_length
        
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = random.choice(self.dataset[idx]['lines'])
        text = get_random_cut(
            line, max_length=self.model_max_length - 2,
            # minus two for the [BOS] and [EOS] tokens
        )

        image, gen_params = generate_line(line)
        
        if self.transform is not None:
            pixels = self.transform(image)
        else:
            pixels = image

        return {
            "pixels": pixels,
            "text": text,
            **gen_params,
        }


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
