try:
    from dataset.line_generator import generate_line
except ImportError:
    from line_generator import generate_line

from torchvision.datasets import VisionDataset
import random


class TextLineDataset(VisionDataset):
    def __init__(self, dataset, transform, model_max_length):
        super().__init__(root=None)
        
        self.dataset = dataset
        self.transform = transform
        self.model_max_length = model_max_length
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        line = random.choice(self.dataset[idx]['lines'])
        text = get_random_cut(
            line,
            # minus two for the [BOS] and [EOS] tokens
            max_length=self.model_max_length - 2,
        )

        image = generate_line(line)
        pixels = self.transform(image)

        return {
            "pixels": pixels,
            "text": text,
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

if __name__ == '__main__':
    from torchvision.transforms import Compose, Resize, Grayscale, ToTensor, Normalize
    from transformers import AutoTokenizer
    import matplotlib.pyplot as plt
    
    # expected format of the dataset
    dummy_dataset = [
        {
            "lines": [
                f"Ceci est un test ààà {i}", 
                f"Ceci est un test ççç {i}", 
                f"Ceci est un test ééé {i}"
            ],
        } for i in range(10000)
    ]
    
    # create a tokenizer
    dummy_tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-cased", 
        model_max_length=100
    )
    
    # create a transform
    transform = Compose([
        Resize((128, 640)),
        Grayscale(),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])

    # create a dataset
    dataset = TextLineDataset(
        dataset=dummy_dataset,
        transform=transform,
        model_max_length=dummy_tokenizer.model_max_length,
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        prefetch_factor=2 * (32 // 8),
        shuffle=True,
    )

    import time
    from tqdm import tqdm
    for batch in tqdm(loader):
        _ = batch
        time.sleep(0.1)